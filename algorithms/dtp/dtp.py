from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from beyond_backprop.algorithms.algo_with_feedback import AlgorithmWithFeedbackNetwork

from beyond_backprop.algorithms.dtp.feedback_networks.utils import (
    feedback_pass,
    forward_pass,
    has_maxpool2d,
)
from beyond_backprop.algorithms.image_classification import ImageClassificationAlgorithm
from beyond_backprop.configs.lr_scheduler import CosineAnnealingLRConfig
from beyond_backprop.configs.optimizer import SGDConfig
from beyond_backprop.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from beyond_backprop.networks.conv_architecture import (
    get_all_forward_activations_and_indices,
)
from beyond_backprop.networks.invertible import set_input_output_shapes_on_forward
from beyond_backprop.networks.layers import Sequential
from beyond_backprop.networks.lenet import LeNet
from beyond_backprop.networks.resnet import ResNet18, ResNet34
from beyond_backprop.networks.simple_vgg import SimpleVGG
from beyond_backprop.utils.hydra_utils import Partial, add_attributes
from beyond_backprop.utils.utils import is_trainable

from ..algorithm import PhaseStr, StepOutputDict
from .feedback_loss import get_feedback_loss
from .feedback_networks.init_symmetric_weights import init_symmetric_weights
from .feedback_networks.pseudoinvert import pseudoinvert
from .metrics import compute_dist_angle
from .utils import make_stacked_feedback_training_figure

logger = logging.getLogger(__name__)
T = TypeVar("T")

DTPNetworkType = TypeVar("DTPNetworkType", LeNet, SimpleVGG, ResNet18, ResNet34)


class DTP(ImageClassificationAlgorithm[DTPNetworkType], AlgorithmWithFeedbackNetwork):
    """Differential Target Propagation algorithm, implemented as a LightningModule."""

    @dataclass
    class HParams(ImageClassificationAlgorithm.HParams):
        """Hyper-Parameters of the model.

        The number of noise samples to use per iteration is set by
        `feedback_samples_per_iteration`.

        NOTE: By increasing the value of `feedback_samples_per_iteration` and setting the value of
        `feedback_training_iterations` to 1 for all layers, we could get something close to a
        "parallel" version of DTP, however the feedback layers are still updated in sequence.
        """

        lr_scheduler: CosineAnnealingLRConfig = field(
            default_factory=lambda: CosineAnnealingLRConfig()
        )
        """Arguments to be passed to the LR scheduler."""

        use_scheduler: bool = True
        """Use of a learning rate scheduler for the forward weights."""

        batch_size: int = 128
        """Batch size."""

        feedback_training_iterations: list[int] = field(default_factory=[20, 30, 35, 55, 20].copy)
        """Number of training steps for the feedback weights per batch.

        Can be a list of integers, where each value represents the number of iterations for that
        layer.
        """

        max_epochs: int = 90
        """Max number of training epochs in total."""

        b_optim: SGDConfig = SGDConfig(
            lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18],
            momentum=0.9,
            weight_decay=0,
        )
        """Hyper-parameters for the optimizer of the feedback weights (backward net)."""

        noise: list[float] = field(default_factory=[0.4, 0.4, 0.2, 0.2, 0.08].copy)
        """The scale of the gaussian random variable in the feedback loss calculation."""

        f_optim: SGDConfig = field(
            default_factory=lambda: SGDConfig(
                lr=0.05,
                momentum=0.9,
                weight_decay=1e-4,
            ),
            # metadata={"omegaconf_ignore": True},
        )
        """Hyper-parameters for the forward optimizer."""

        beta: float = 0.7
        """nudging parameter: Used when calculating the first target."""

        feedback_samples_per_iteration: int = 1
        """Number of noise samples to use to get the feedback loss in a single iteration.

        NOTE: The loss used for each update is the average of these losses.
        """

        early_stopping_patience: int = 0
        """Max number of epochs to train for without an improvement to the validation accuracy
        before the training is stopped.

        When 0, no early stopping is used.
        """

        init_symetric_weights: bool = False
        """Sets symmetric weight initialization.

        Useful for debugging.
        """

        # TODO: Add a Callback class to compute and plot jacobians, if that's interesting.
        # jacobian: bool = False  # compute jacobians

        plot_every: int = 1000
        """Step interval for creating and logging plots."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: DTPNetworkType,
        hp: DTP.HParams,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)

        self.forward_network: nn.Module = self.make_forward_network(base_network=network)
        self.feedback_network: nn.Module = self.make_feedback_network(
            base_network=network, forward_network=self.forward_network
        )

        # NOTE: These properties below are in the backward ordering, while those in the hparams are
        # in the forward order. This is so they line up with the trainable layers of
        # self.backward_net, which is in the backward order.
        self.hp: DTP.HParams
        # The number of iterations to perform for each of the layers in `self.backward_net`.
        self.feedback_iterations = self._align_values_with_backward_net(
            self.hp.feedback_training_iterations,
            default=0,
            inputs_are_forward_ordered=True,
        )
        # The noise scale for each feedback layer.
        self.feedback_noise_scales = self._align_values_with_backward_net(
            self.hp.noise,
            default=0.0,
            inputs_are_forward_ordered=True,
        )

        # The learning rate for each feedback layer.
        # NOTE: We use "Partial" so we can fetch the value of a parater in the partial with an
        # attribute lookup, e.g.
        self.make_b_optim: Partial[Optimizer] = add_attributes(instantiate(self.hp.b_optim))
        self.make_f_optim: Partial[Optimizer] = add_attributes(instantiate(self.hp.f_optim))

        lrs_per_layer = self.make_b_optim.lr
        self.feedback_lrs = self._align_values_with_backward_net(
            lrs_per_layer, default=0.0, inputs_are_forward_ordered=True
        )

        print("Feedback net:")
        print(self.feedback_network)
        # Check that the hparams line up correctly with the trainable layers of the backward
        # net.
        _validate_hparam_configuration(model=self)

        # Can't do automatic optimization here, since we do multiple sequential updates
        # per batch.
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self._feedback_optimizers: list[Optimizer | None] | None = None
        self._forward_optimizer: Optimizer | None = None

        self.save_hyperparameters(ignore=["network"])

    def make_forward_network(self, base_network: DTPNetworkType) -> DTPNetworkType:
        """Adapt the base network to create the forward net used by this algorithm.

        In the case of DTP, we assume we don't change anything about the base network and use it
        as-is. Errors might get raised when creating the feedback network if some layer isn't
        supported.
        """
        # NOTE:
        if has_maxpool2d(base_network):
            # GitHub co-pilot's idea, seems pretty interesting! Could DTP Replace MaxPool with
            # AvgPool to create the feedback net?
            # self.forward_net.apply(set_maxpool2d_to_avgpool2d)

            def set_return_indices(m: nn.Module) -> None:
                if isinstance(m, nn.MaxPool2d):
                    m.return_indices = True

            # TODO: this is a bit too specific to my taste.
            assert isinstance(base_network, (LeNet, SimpleVGG))
            base_network._return_maxpool_indices = True
            base_network.apply(set_return_indices)

        return base_network

    def make_feedback_network(
        self, base_network: DTPNetworkType, forward_network: nn.Module
    ) -> nn.Sequential:
        # Pass an example input through the forward net so that we know the input/output shapes for
        # each layer. This makes it easier to then create the feedback (a.k.a backward) net.
        set_input_output_shapes_on_forward(forward_network)

        example_out, maxpool_indices = forward_pass(forward_network, self.example_input_array)
        assert example_out.requires_grad
        # Get the "pseudo-inverse" of the forward network:

        feedback_network = pseudoinvert(forward_network).to(self.device)  # type: ignore
        # Pass the output of the forward net for the `example_input_array` through the
        # backward net, to check that the backward net is indeed able to recover the
        # inputs (at least in terms of their shape for now).
        example_in_hat = feedback_pass(feedback_network, example_out, maxpool_indices)

        assert example_in_hat.requires_grad
        assert example_in_hat.shape == self.example_input_array.shape
        assert example_in_hat.dtype == self.example_input_array.dtype

        if self.hp.init_symetric_weights:
            logger.info("Initializing the backward net with symmetric weights.")
            init_symmetric_weights(forward_network, feedback_network)

        feedback_network = feedback_network.to(self.device)
        return feedback_network

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        output, max_indices = forward_pass(self.forward_network, input)
        feedback_pass(self.feedback_network, output, max_indices)
        return output

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        """Main step, used by the `[training/valid/test]_step` methods."""
        x, y = batch

        # ----------- Optimize the feedback weights -------------
        # NOTE: feedback_loss here returns a dict for now, since I think that makes things easier
        # to inspect.
        feedback_training_outputs: dict = self.feedback_loss(x, y, phase=phase)

        feedback_loss: Tensor = feedback_training_outputs["loss"]
        avg_feedback_loss: Tensor = feedback_training_outputs["avg_loss"]
        if self.trainer is not None:
            self.log(f"{phase}/B_loss", feedback_loss)
            self.log(f"{phase}/B_avg_loss", avg_feedback_loss, prog_bar=phase == "train")
        # This is never a 'live' loss, since we do the optimization steps sequentially
        # inside `feedback_loss`.
        assert not feedback_loss.requires_grad

        # ----------- Optimize the forward weights -------------
        forward_training_outputs: dict = self.forward_loss(x, y, phase=phase)
        forward_loss: Tensor = forward_training_outputs["loss"]

        # During training, the forward loss will be a 'live' loss tensor, since we
        # gather the losses for each layer. Here we perform only one step.
        assert forward_loss.requires_grad == (phase == "train")
        # NOTE: If this is getting called from the `ParallelDTP`, then
        # `self.automatic_optimization` will be `True`, and we let PL do the update.
        if not self.automatic_optimization and forward_loss.requires_grad:
            forward_optimizer = self.forward_optimizer
            forward_optimizer.zero_grad()
            if self.trainer:
                self.manual_backward(forward_loss)
            else:
                # Unit testing.
                forward_loss.backward()
            forward_optimizer.step()
            self.log("F_lr", forward_optimizer.param_groups[0]["lr"], sync_dist=True)
            forward_loss = forward_loss.detach()

        last_layer_loss: Tensor = forward_training_outputs["layer_losses"][-1].detach()
        if self.trainer is not None:
            self.log(f"{phase}/F_loss", forward_loss, sync_dist=True)
            self.log(f"{phase}/Loss", last_layer_loss, prog_bar=phase == "train", sync_dist=True)

        # Since here we do manual optimization, we just return a float. This tells PL that we've
        # already performed the optimization steps, if needed.
        logits = forward_training_outputs["logits"]
        total_loss = forward_loss + feedback_loss
        # NOTE: These logits shouldn't require grad anymore.
        assert not logits.requires_grad
        return {"loss": total_loss, "y": y, "logits": logits}

    def feedback_loss(self, x: Tensor, y: Tensor, phase: str) -> dict[str, Any]:
        n_layers = len(self.feedback_network)
        # Reverse the backward net, just for ease of readability.
        reversed_backward_net = self.feedback_network[::-1]
        reversed_feedback_optimizers = self.feedback_optimizers()[::-1]
        # Also reverse these values so they stay aligned with the net above.
        noise_scale_per_layer = list(reversed(self.feedback_noise_scales))
        iterations_per_layer = list(reversed(self.feedback_iterations))

        # NOTE: We never train the last layer of the feedback net (G_0).
        assert iterations_per_layer[0] == 0
        assert noise_scale_per_layer[0] == 0

        # NOTE: We can compute all the ys for all the layers up-front, because we don't
        # update the forward weights.
        # 1- Compute the forward activations (no grad).
        with torch.no_grad():
            ys: list[Tensor]
            maxpool_indices: list[Tensor]

            ys, maxpool_indices = get_all_forward_activations_and_indices(
                self.forward_network, x, allow_grads_between_layers=False
            )

        # List of losses, distances, and angles for each layer (with multiple iterations per
        # layer).
        layer_losses: list[list[Tensor]] = []
        layer_angles: list[list[float]] = []
        layer_distances: list[list[float]] = []
        layer_avg_losses: list[list[float]] = []

        # Layer-wise autoencoder training begins:
        # NOTE: Skipping the first layer
        for layer_index in range(1, n_layers):
            # Forward layer
            F_i = self.forward_network[layer_index]
            # Feedback layer
            G_i = reversed_backward_net[layer_index]
            layer_optimizer = reversed_feedback_optimizers[layer_index]
            assert (layer_optimizer is not None) == self._is_trainable(G_i)

            x_i = ys[layer_index - 1]
            y_i = ys[layer_index]
            if layer_index < len(maxpool_indices):
                maxpool_indices_i = maxpool_indices[layer_index]
            else:
                maxpool_indices_i = None

            # Number of feedback training iterations to perform for this layer.
            iterations_i = iterations_per_layer[layer_index]
            if iterations_i and not self.training:
                # NOTE: Only perform one iteration per layer when not training.
                iterations_i = 1
            # The scale of noise to use for this layer.
            noise_scale_i = noise_scale_per_layer[layer_index]

            # Collect the distances and angles between the forward and backward weights during this
            # iteratin.
            iteration_angles: list[float] = []
            iteration_distances: list[float] = []
            iteration_losses: list[Tensor] = []

            # NOTE: When a layer isn't trainable (e.g. layer is a Reshape or nn.ELU), then
            # iterations_i will be 0, so the for loop below won't be run.

            # Train the current autoencoder:
            for iteration in range(iterations_i):
                assert noise_scale_i > 0, (
                    layer_index,
                    iterations_i,
                )
                # Get the loss (see `feedback_loss.py`)
                loss = self.layer_feedback_loss(
                    feedback_layer=G_i,
                    forward_layer=F_i,
                    input=x_i,
                    output=y_i,
                    maxpool_indices=maxpool_indices_i,
                    noise_scale=noise_scale_i,
                    noise_samples=self.hp.feedback_samples_per_iteration,
                )

                # Compute the angle and distance for debugging the training of the
                # feedback weights:
                with torch.no_grad():
                    metrics = compute_dist_angle(F_i, G_i)
                    if isinstance(metrics, dict):
                        # NOTE: When a block has more than one trainable layer, we only report the
                        # first non-zero value for now.
                        # TODO: Fix this later.
                        distance, angle = 0, 0
                        for k, v in metrics.items():
                            if v != (0, 0):
                                if isinstance(v, dict):
                                    # Skip this layer.
                                    continue
                                distance, angle = v
                                break
                    else:
                        distance, angle = metrics

                # perform the optimization step for that layer when training.
                if self.training and layer_optimizer:
                    assert isinstance(loss, Tensor) and loss.requires_grad
                    layer_optimizer.zero_grad()
                    # self.manual_backward(loss) won't work if self.trainer is None
                    # self.trainer is None in legacy unit tests
                    self.manual_backward(loss) if self.trainer is not None else loss.backward()
                    layer_optimizer.step()
                    if self.trainer is not None:
                        self.log(
                            f"B_lr{layer_index}",
                            layer_optimizer.param_groups[0]["lr"],
                            sync_dist=True,
                        )
                    loss = loss.detach()
                else:
                    assert isinstance(loss, Tensor) and not loss.requires_grad
                    # When not training that layer,
                    loss = torch.as_tensor(loss, device=y.device)

                logger.debug(
                    f"Layer {layer_index}, Iteration {iteration}/{iterations_i}, angle={angle}, "
                    f"distance={distance}, loss={loss}"
                )
                iteration_losses.append(loss)
                iteration_angles.append(angle)
                iteration_distances.append(distance)

                # IDEA: If we log these values once per iteration, will the plots look nice?
                # self.log(f"{self.phase}/B_loss[{layer_index}]", loss)
                # self.log(f"{self.phase}/B_angle[{layer_index}]", angle)
                # self.log(f"{self.phase}/B_distance[{layer_index}]", distance)

            layer_losses.append(iteration_losses)
            layer_angles.append(iteration_angles)
            layer_distances.append(iteration_distances)

            # IDEA: Logging the number of iterations could be useful if we add some kind of early
            # stopping for the feedback training, since the number of iterations might vary.
            total_iter_loss = sum(iteration_losses)
            if iterations_i > 0:
                avg_iter_loss = total_iter_loss / iterations_i
                layer_avg_losses.append(avg_iter_loss)

            if self.trainer is not None:
                self.log(f"{phase}/B_total_loss[{layer_index}]", total_iter_loss)
                if iterations_i > 0:
                    self.log(f"{phase}/B_avg_loss[{layer_index}]", avg_iter_loss)
                self.log(f"{phase}/B_iterations[{layer_index}]", float(iterations_i))
                self.log(f"{phase}/B_angle[{layer_index}]", iteration_angles[-1])
                self.log(f"{phase}/B_distance[{layer_index}]", iteration_distances[-1])

        if (
            self.training
            and self.global_step % self.hp.plot_every == 0
            and self.trainer is not None
        ):
            fig = make_stacked_feedback_training_figure(
                all_values=[layer_angles, layer_distances, layer_losses],
                row_titles=["angles", "distances", "losses"],
                title_text=(
                    "Evolution of various metrics during feedback weight training "
                    f"(global_step={self.global_step})"
                ),
            )
            fig_name = f"feedback_training_{self.global_step}"
            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except ValueError:
                output_dir = self.trainer.log_dir or "."

            figures_dir = Path(output_dir) / "figures"
            figures_dir.mkdir(exist_ok=True, parents=False)

            save_path: Path = figures_dir / fig_name
            fig.write_image(str(save_path.with_suffix(".png")))
            logger.info(f"Figure saved at path {save_path.with_suffix('.png')}")

            # FIXME: Re-enable perhaps?
            # TODO: Find a better way to detect if we're in 'debug' mode.
            if logger.level == logging.DEBUG:
                # Also save an HTML version when debugging.
                fig.write_html(str(save_path.with_suffix(".html")), include_plotlyjs="cdn")

            if wandb.run:
                wandb.log({"feedback_training": fig})

        # NOTE: Need to return something.
        total_b_loss = sum(sum(iteration_losses) for iteration_losses in layer_losses)
        avg_b_loss = sum(layer_avg_losses) / len(layer_avg_losses)
        return {
            "loss": total_b_loss,
            "avg_loss": avg_b_loss,
            "layer_losses": layer_losses,
            "layer_angles": layer_angles,
            "layer_distances": layer_distances,
        }

    def forward_loss(self, x: Tensor, y: Tensor, phase: str) -> dict[str, Tensor | Any]:
        """Get the loss used to train the forward net.

        NOTE: Unlike `feedback_loss`, this actually returns the 'live' loss tensor.
        """
        # NOTE: Sanity check: Use standard backpropagation for training rather than TP.
        # --------
        # return super().forward_loss(x=x, y=y)
        # --------
        ys, maxpool_indices = get_all_forward_activations_and_indices(
            self.forward_network,
            x,
            allow_grads_between_layers=False,
        )
        logits = ys[-1]
        labels = y

        # Calculate the first target using the gradients of the loss w.r.t. the logits.
        # NOTE: Need to manually enable grad here so that we can also compute the first
        # target during validation / testing.
        with torch.inference_mode(False), torch.set_grad_enabled(True):
            temp_logits = logits.detach().clone().requires_grad_(True)
            # NOTE: This is necessary when doing validation since torch.inference_mode was set
            # when this tensor was created.
            if labels.is_inference():
                labels = labels.clone()
            ce_loss = F.cross_entropy(temp_logits, labels, reduction="sum")
            grads = torch.autograd.grad(
                outputs=ce_loss,
                inputs=temp_logits,
                only_inputs=True,  # Do not backpropagate further than the input tensor!
                create_graph=False,
            )
            assert len(grads) == 1

        y_n_grad = grads[0]
        delta = -self.hp.beta * y_n_grad

        if self.trainer is not None:
            self.log(f"{phase}/delta.norm()", delta.norm())
        # Compute the first target (for the last layer of the forward network):
        last_layer_target = logits.detach() + delta

        N = len(self.forward_network)
        # NOTE: Initialize the list of targets with Nones, and we'll replace all the
        # entries with tensors corresponding to the targets of each layer.
        targets: list[Tensor | None] = [
            *(None for _ in range(N - 1)),
            last_layer_target,
        ]

        # Reverse the ordering of the layers, just to make the indexing in the code below match
        # those of the math equations.
        reordered_feedback_net: Sequential = self.feedback_network[::-1]  # type: ignore

        # Calculate the targets for each layer, moving backward through the forward net:
        # N-1, N-2, ..., 2, 1
        # NOTE: Starting from N-1 since we already have the target for the last layer).
        with torch.no_grad():
            for i in reversed(range(1, N)):
                G = reordered_feedback_net[i]
                # G = feedback_net[-1 - i]

                assert targets[i - 1] is None  # Make sure we're not overwriting anything.
                # NOTE: Shifted the indices by 1 compared to @ernoult's eq.
                # t^{n-1} = s^{n-1} + G(t^{n}; B) - G(s^{n} ; B).
                # targets[i - 1] = ys[i - 1] + G(targets[i]) - G(ys[i])
                prev_target = targets[i]
                assert prev_target is not None
                targets[i - 1] = self.compute_target(
                    i=i, G=G, hs=ys, prev_target=prev_target, maxpool_indices=maxpool_indices
                )

                # NOTE: Alternatively, just target propagation:
                # targets[i - 1] = G(targets[i])

        # NOTE: targets[0] is the targets for the output of the first layer, not for x.
        # Make sure that all targets have been computed and that they are fixed (don't require
        # grad)
        assert all(target is not None and not target.requires_grad for target in targets)

        # Calculate the losses for each layer:
        forward_loss_per_layer = []
        for i in range(0, N):
            if (
                ys[i].requires_grad or phase != "train"
            ):  # Removes duplicate reshape layer loss from total loss estimate
                layer_loss = (
                    0.5 * ((ys[i] - targets[i]) ** 2).view(ys[i].size(0), -1).sum(1).mean()
                )
                # NOTE: Apparently NOT Equivalent to the following!
                # 0.5 * F.mse_loss(ys[i], target_tensors[i], reduction="mean")
                forward_loss_per_layer.append(layer_loss)

        # self.trainer is None in some unit tests which only use PL module
        if self.trainer is not None:
            for i, layer_loss in enumerate(forward_loss_per_layer):
                self.log(f"{phase}/F_loss[{i}]", layer_loss)

        loss_tensor = torch.stack(forward_loss_per_layer, dim=0)
        # TODO: Use 'sum' or 'mean' as the reduction between layers?
        forward_loss = loss_tensor.sum(dim=0)
        return {
            "loss": forward_loss,
            "layer_losses": forward_loss_per_layer,
            "logits": logits.detach(),
        }

    def compute_target(
        self,
        i: int,
        G: nn.Module,
        hs: list[Tensor],
        prev_target: Tensor,
        maxpool_indices: list[Tensor] | None = None,
    ) -> Tensor:
        """Compute the target of the previous forward layer. given , the associated feedback layer,
        the activations for each layer, and the target of the current layer.

        Parameters
        ----------
        i : int
            the index of the forward layer for which we want to compute a target
        G : nn.Module
            the associated feedback layer
        hs : List[Tensor]
            the activations for each layer
        prev_target : Tensor
            The target of the next forward layer.
        maxpool_indices: list[Tensor] | None
            The max indices from the nn.MaxPool2d operations of each convolutional block in the
            forward net.

        Returns
        -------
        Tensor
            The target to use to train the forward layer at index `i`.
        """
        if not maxpool_indices:
            return hs[i - 1] + G(prev_target) - G(hs[i])

        from .feedback_networks.conv_feedback_network import ConvFeedbackBlock

        if i in range(len(maxpool_indices)):
            assert isinstance(G, ConvFeedbackBlock)
            # FIXME: Not quite right.
            return hs[i - 1] + G(prev_target, maxpool_indices[i]) - G(hs[i], maxpool_indices[i])

        return hs[i - 1] + G(prev_target) - G(hs[i])

    def layer_feedback_loss(
        self,
        *,
        feedback_layer: nn.Module,
        forward_layer: nn.Module,
        input: Tensor,
        output: Tensor,
        maxpool_indices: Tensor | None = None,
        noise_scale: float | Tensor,
        noise_samples: int = 1,
    ) -> Tensor:
        return get_feedback_loss(
            feedback_layer=feedback_layer,
            forward_layer=forward_layer,
            input=input,
            output=output,
            maxpool_indices=maxpool_indices,
            noise_scale=noise_scale,
            noise_samples=noise_samples,
        )

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler and not self.automatic_optimization:
            assert not isinstance(lr_scheduler, list)
            lr_scheduler.step()

    def configure_optimizers(self) -> list[dict]:
        """Returns the configurations for the optimizers.

        The entries are ordered like: [G_N, G_N-1, ..., G_2, G_1, F]

        The first items in the list are the configs for each trainable feedback layer.
        There is no entry for the first feedback layer (G_0)
        The last item in the list is for the forward optimizer.
        """
        # NOTE: We pass the learning rates in the same order as the feedback net:
        # NOTE: Here when we have one optimizer per feedback layer, we will put the forward optim
        # at the last index.
        configs: list[dict] = []
        # NOTE: The last feedback layer (G_0) isn't trained, so it doesn't have an optimizer.
        assert len(self.feedback_network) == len(self.feedback_lrs)
        for i, (feedback_layer, lr) in enumerate(zip(self.feedback_network, self.feedback_lrs)):
            if i == (len(self.feedback_network) - 1) or not is_trainable(feedback_layer):
                # NOTE: No learning rate for the first feedback layer atm, although we very well
                # could train it, it wouldn't change anything about the forward weights.
                # Non-trainable layers also don't have an optimizer.
                assert lr == 0.0
            else:
                assert lr != 0.0
                # TODO: Fix this.
                feedback_layer_optimizer = self.make_b_optim(feedback_layer.parameters(), lr=lr)
                feedback_layer_optim_config = {"optimizer": feedback_layer_optimizer}
                configs.append(feedback_layer_optim_config)

        # Forward optimizer:
        forward_optimizer = self.make_f_optim(self.forward_network.parameters())
        forward_optim_config: dict[str, Any] = {"optimizer": forward_optimizer}
        if self.hp.use_scheduler:
            # Using the same LR scheduler as the original code:

            make_lr_scheduler = instantiate(self.hp.lr_scheduler)
            lr_scheduler = make_lr_scheduler(forward_optimizer)
            lr_scheduler_config: dict[str, Any] = {"scheduler": lr_scheduler}
            if self.automatic_optimization:
                lr_scheduler_config.update(
                    {
                        "interval": self.hp.lr_scheduler.interval,
                        "frequency": self.hp.lr_scheduler.frequency,
                    }
                )
            forward_optim_config["lr_scheduler"] = lr_scheduler_config
        configs.append(forward_optim_config)
        return configs

    def _is_trainable(self, layer: nn.Module) -> bool:
        # TODO: is_trainable should be working, but with the DistributedDataParallel wrapper, seems
        # like it's not always working quite right.
        # if layer in self.backward_net:
        #     layer_index = {
        #         i for i, m in enumerate(self.backward_net) if m is layer
        #     }.pop()
        return is_trainable(layer)

    def feedback_optimizers(self) -> list[Optimizer | None]:
        """Returns the list of optimizers, one per layer of the feedback/backward net:

        [G_N, G_N-1, ..., G_2, G_1, None]

        For the "first" feedback layer (G_0), as well as all layers without trainable weights, the
        entry will be `None`.
        """
        # NOTE: self.trainer is None during unit testing
        if self.trainer is None:
            return self._feedback_optimizers
        elif hasattr(self, "_feedback_optimizers") and self._feedback_optimizers is not None:
            return self._feedback_optimizers

        _feedback_optimizers = []
        # NOTE: G[0] layer is not currently trained (although it could eventually if
        # we wanted an end-to-end invertible network).
        # Go until the penultimate layer of the backward net.
        optimizers = list(self.optimizers())
        for i, layer in enumerate(self.feedback_network[:-1]):
            layer_optimizer: Optimizer | None = None
            if self._is_trainable(layer):
                layer_optimizer = optimizers.pop(0)
            _feedback_optimizers.append(layer_optimizer)
        _feedback_optimizers.append(None)
        # Only one left: The optimizer for the forward net.
        # BUG: This here doesn't work when using multiple GPUs by default. Seems to be caused by
        # is_trainable always returning False, which might be caused by some
        # DistributedDataParallel wrapper or something like that.
        assert len(optimizers) == 1, optimizers

        assert optimizers[-1] is self.forward_optimizer
        self._feedback_optimizers = _feedback_optimizers
        return _feedback_optimizers

    @property
    def forward_optimizer(self) -> Optimizer:
        """Returns The optimizer of the forward net."""
        # NOTE: self.trainer is None during unit testing
        if self.trainer is None:
            return self._forward_optimizer
        return self.optimizers()[-1]

    def _align_values_with_backward_net(
        self, values: list[T], default: T, inputs_are_forward_ordered: bool = False
    ) -> list[T]:
        """Aligns the values in `values` so that they are aligned with the trainable layers in the
        backward net. The last layer of the backward net (G_0) is also never trained.

        This assumes that `forward_ordering` is True, then `values` are forward-ordered.
        Otherwise, assumes that the input is given in the *backward* order Gn, Gn-1, ..., G0.

        NOTE: Outputs are *always* aligned with `self.backward_net` ([Gn, ..., G0]).

        Example: Using the default learning rate values for cifar10 as an example:

            `self.forward_net`: (conv, conv, conv, conv, reshape, linear)
            `self.backward_net`:   (linear, reshape, conv, conv, conv, conv)

            forward-aligned values: [1e-4, 3.5e-4, 8e-3, 8e-3, 0.18]

            `values` (backward-aligned): [0.18, 8e-3, 8e-3, 3.5e-4, 1e-4]  (note: backward order)


            `default`: 0

            Output:  [0.18, 0 (default), 8e-3, 8e-3, 3.5e-4, 1e-4, 0 (never trained)]

        Parameters
        ----------
        values : List[T]
            List of values for each trainable layer.
        default : T
            The value to set for non-trainable layers in the feedback network.
        inputs_are_forward_ordered : bool, optional
            Whether the inputs are given in a forward-aligned order or not. When they aren't, they
            are aligned with the trainable layers in the feedback network.
            Defaults to False.

        Returns
        -------
        List[T]
            List of values, one per layer in the backward net, with the non-trainable layers
            assigned the value of `default`.
        """
        # FIXME: Fix this for the new LeNetFeedbackNetwork class: Need to flatten the network a
        # bit.

        n_layers_that_need_a_value = sum(map(is_trainable, self.feedback_network))
        # Don't count the last layer of the backward net (i.e. G_0), since we don't
        # train it.
        n_layers_that_need_a_value -= 1

        if isinstance(values, (int, float)):
            values = [values for _ in range(n_layers_that_need_a_value)]  # type: ignore

        if len(values) > n_layers_that_need_a_value:
            truncated_values = values[:n_layers_that_need_a_value]
            # TODO: Eventually, either create a parameterized default value for the HParams for
            # each dataset, or switch to Hydra, if it's easy to use and solves this neatly.
            # warnings.warn(
            raise RuntimeError(
                f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
                f"given {len(values)} values! (values={values})\n"
                f"Either pass a single value for all layers, or a value for each layer.\n"
                # f"WARNING: This will only use the first {n_layers_that_need_a_value} values: "
                # f"{truncated_values}"
            )
            values = truncated_values
        elif len(values) < n_layers_that_need_a_value:
            # TODO: Same as above.
            last_value = values[-1]
            raise RuntimeError(
                # warnings.warn(
                # RuntimeWarning(
                f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
                f"only provided {len(values)} values! (values={values})\n"
                f"Either pass a single value for all layers, or a value for each layer.\n"
                f"WARNING: This will duplicate the first value ({last_value}) for all remaining "
                f"layers."
            )
            values = list(values) + [
                last_value for _ in range(n_layers_that_need_a_value - len(values))
            ]
            logger.warn(f"New values: {values}")
            assert len(values) == n_layers_that_need_a_value

        backward_ordered_input = list(reversed(values)) if inputs_are_forward_ordered else values
        # TODO: Once above TODO is addressed, re-instate this policy.
        # if len(values) != n_layers_that_need_a_value:
        #     raise ValueError(
        #         f"There are {n_layers_that_need_a_value} layers that need a value, but we were "
        #         f"given {len(values)} values! (values={values})\n "
        #     )

        values_left = backward_ordered_input.copy()
        values_per_layer: list[T] = []
        for layer in self.feedback_network:
            if is_trainable(layer) and values_left:
                values_per_layer.append(values_left.pop(0))
            else:
                values_per_layer.append(default)
        assert values_per_layer[-1] == default

        backward_ordered_output = values_per_layer
        assert all(
            isinstance(v, (int, float)) for v in backward_ordered_output
        ), backward_ordered_output
        return backward_ordered_output

    @property
    def is_multi_gpu(self) -> bool:
        if self.trainer.devices == -1:
            return torch.cuda.device_count() > 1
        return self.trainer.devices != 1 or (
            isinstance(self.trainer.devices, list) and len(self.trainer.devices) > 1
        )


def _validate_hparam_configuration(model: DTP) -> None:
    """Check that the hparams line up correctly with the trainable layers of the backward net."""
    N = len(model.feedback_network)
    for i, (layer, lr, noise, iterations) in list(
        enumerate(
            zip(
                model.feedback_network,
                model.feedback_lrs,
                model.feedback_noise_scales,
                model.feedback_iterations,
            )
        )
    ):
        logger.info(
            f"self.backward_net[{i}]: (G[{N-i-1}]): Type {type(layer).__name__}, LR: {lr}, "
            f"noise: {noise}, iterations: {iterations}"
        )
        if i == N - 1:
            # The last layer of the backward_net (the layer closest to the input) is not
            # currently being trained, so we expect it to not have these parameters.
            assert lr == 0
            assert noise == 0
            assert iterations == 0
            continue
        if any(p.requires_grad for p in layer.parameters()):
            # For any of the trainable layers in the backward net (except the last one), we
            # expect to have positive values:
            assert lr > 0
            assert noise > 0
            assert iterations > 0
        else:
            # Non-Trainable layers (e.g. Reshape) are not trained.
            assert lr == 0
            assert noise == 0
            assert iterations == 0
