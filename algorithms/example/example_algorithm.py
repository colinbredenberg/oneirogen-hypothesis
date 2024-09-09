from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable, TypeVar

import torch
from hydra_zen import instantiate
from lightning import Callback
from torch import Tensor, nn
from torch.nn import functional as F

from beyond_backprop.configs.optimizer import AdamConfig
from beyond_backprop.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from beyond_backprop.networks import LeNet, SimpleVGG
from beyond_backprop.networks.invertible import set_input_output_shapes_on_forward
from beyond_backprop.networks.layers import Sequential, get_all_forward_activations
from beyond_backprop.networks.network import Network
from beyond_backprop.utils.types import PhaseStr, StepOutputDict
from beyond_backprop.utils.utils import tile_batch
from beyond_backprop.algorithms.image_classification import ImageClassificationAlgorithm

from ..algorithm import Algorithm
from .feedback_network import create_feedback_network
from .forward_network import create_forward_network

logger = get_logger(__name__)

# Type variable that describes the supported base networks. This is just a fancy way of showing at
# 'compile-time' that the base network must be one of the supported types, otherwise an error will
# be raised in the `make_forward_network` method.
SupportedNetworkType = TypeVar("SupportedNetworkType", LeNet, SimpleVGG)


class ExampleAlgorithm(ImageClassificationAlgorithm[SupportedNetworkType]):
    """Example of an algorithm that uses a Difference-Target-Propagation (DTP)-like approach.

    This is meant to serve as an illustration of how to organize your code for a new algorithm.
    This isn't exactly equivalent to the DTP algorithm.

    Notes:
    - This also gives an example of how the algorithm can adapt the base network however it wants
      to create the `forward_network` it uses for the forward pass.
    - The `feedback_network` is created from the `forward_network` using a dynamic dispatch
      function, which is very flexible. However, we limit the applicability of this algorithm only
      to the LeNet and SimpleVGG architectures, even though the same approach could be used for
      ResNets or other architectures.
    """

    @dataclass
    class HParams(Algorithm.HParams):
        """Hyper-parameters of the algorithm."""

        forward_optim: AdamConfig = AdamConfig(
            lr=0.001,
            # momentum=0.9,
            weight_decay=1e-4,
        )
        """Parameters for the optimizer used to train the forward network."""

        feedback_optim: AdamConfig = AdamConfig(
            lr=0.01,
            # lr=[1e-4, 3.5e-4, 8e-3, 8e-3, 0.18],
            # momentum=0.9,
            weight_decay=1e-4,
        )
        """Parameters for the optimizer used to train the feedback network."""

        # TODO: Show how to add LR schedulers.
        # forward_lr_scheduler: LRSchedulerConfig | None = None
        # feedback_lr_scheduler: LRSchedulerConfig | None = None

        feedback_noise_std: float = 0.05
        """Std dev of the noise used to perturb the layer inputs and outputs in fb training."""

        n_noise_samples: int = 10
        """Number of noise samples to use when calculating the feedback loss."""

        beta: float = 0.7
        """nudging parameter: Used when calculating the first target."""

        feedback_training_iterations: int = 10
        """number of feedback training iterations per forward training iteration."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: SupportedNetworkType,
        hp: ExampleAlgorithm.HParams | None = None,
    ):
        super().__init__(datamodule, network, hp)
        self.hp: ExampleAlgorithm.HParams
        self.forward_network = self.make_forward_network(network)
        self.feedback_network = self.make_feedback_network(network, self.forward_network)

    def make_forward_network(self, base_network: Network) -> Sequential:
        """Creates the forward network by adapting the base network."""
        assert isinstance(base_network, nn.Module)
        forward_net = create_forward_network(base_network)
        forward_net.to(self.device)
        return forward_net

    def make_feedback_network(
        self, base_network: Network, forward_network: Sequential
    ) -> Sequential:
        """Creates the feedback network based on the base and forward networks."""
        # NOTE: In this example here, we only need the forward network to create the feedback
        # network.

        set_input_output_shapes_on_forward(forward_network)
        was_training = forward_network.training
        with torch.no_grad():
            forward_network.eval()
            dummy_output = forward_network(self.example_input_array)
            forward_network.train(was_training)

        feedback_network = create_feedback_network(forward_network)
        feedback_network.to(self.device)
        reconstructed_input = feedback_network(dummy_output)
        assert reconstructed_input.shape == self.example_input_array.shape
        return feedback_network

    def configure_optimizers(self):
        feedback_optim = self._create_feedback_optimizer()
        forward_optim = self._create_forward_optimizer()
        if self.hp.feedback_training_iterations == 1:
            return [feedback_optim, forward_optim]
        # NOTE: Here I'm using a trick so that we can do multiple feedback training iterations
        # on a single batch and still use PyTorch-Lightning automatic optimization:
        return [
            *(feedback_optim for _ in range(self.hp.feedback_training_iterations)),
            forward_optim,
        ]

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        # Dummy forward pass, not really used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        output = detached_forward(self.forward_network, input)
        reconstructed_input = detached_forward(self.feedback_network, output)
        return output, reconstructed_input

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int
    ) -> StepOutputDict:
        return self.shared_step(batch, batch_idx, optimizer_idx=optimizer_idx, phase="train")

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        phase: PhaseStr,
        optimizer_idx: int | None = None,
    ) -> StepOutputDict:
        """Calculates the loss for the current optimizer for the given batch.

        Returns a dictionary with some items that are used by the base class.
        """
        x, y = batch
        training = phase == "train"

        batch_loss = 0.0

        if optimizer_idx in range(self.hp.feedback_training_iterations) or optimizer_idx is None:
            fb_train_iter = optimizer_idx or 0
            feedback_layer_losses = self.feedback_loss(x=x, y=y, phase=phase)
            feedback_loss = sum(feedback_layer_losses)

            for layer_index, layer_loss in enumerate(feedback_layer_losses):
                self.log(f"{phase}/feedback_loss[{layer_index}][{fb_train_iter}]", layer_loss)
            self.log(f"{phase}/feedback_loss", feedback_loss, logger=True, prog_bar=training)

            batch_loss += feedback_loss

        if optimizer_idx in (self.hp.feedback_training_iterations, None):
            forward_layer_losses = self.forward_loss(x=x, y=y, phase=phase)
            forward_loss = sum(forward_layer_losses)

            for layer_index, layer_loss in enumerate(forward_layer_losses):
                self.log(f"{phase}/forward_loss[{layer_index}]", layer_loss)
            self.log(f"{phase}/forward_loss", forward_loss, logger=True, prog_bar=training)

            batch_loss += forward_loss

        with torch.no_grad():
            logits = self.forward_network(x)
            # Useful when debugging:
            # y_pred = logits.argmax(dim=-1)
            # unique_outputs, output_counts = torch.unique(y_pred, return_counts=True)
            # unique_y, y_counts = torch.unique(y, return_counts=True)
            # logger.debug(
            #     "Outputs: "
            #     + str(dict(zip(unique_outputs.tolist(), output_counts.tolist())))
            #     + "True class labels: "
            #     + str(dict(zip(unique_y.tolist(), y_counts.tolist()))),
            # )

        return StepOutputDict(logits=logits, y=y, loss=batch_loss, log={})

    def feedback_loss(self, x: Tensor, y: Tensor, phase: PhaseStr) -> list[Tensor]:
        """Compute the feedback loss."""
        with torch.no_grad():
            forward_activations = get_all_forward_activations(self.forward_network, x)
        # TODO: Brush-up on the target propagation algorithm. As far as I remember, it goes
        # something like this: Incentivize the feedback network to reconstruct the input (simple
        # reconstruction loss), but also incentivize it to reconstruct the differences between
        # inputs
        layer_losses: list[Tensor] = []
        for i, (forward_layer_i, feedback_layer_i) in enumerate(
            zip(self.forward_network, reversed(self.feedback_network))
        ):
            assert isinstance(forward_layer_i, nn.Module)
            assert isinstance(feedback_layer_i, nn.Module)
            # NOTE: Training the first layer of the feedback network isn't necessary, since it
            # isn't used. However since all the losses are independent, we can just train it if we
            # want. Doesn't really matter.
            # Use this to skip training the first layer of the feedback network:
            if i == 0:
                continue

            layer_input = forward_activations[i - 1] if i > 0 else x
            layer_output = forward_activations[i]

            layer_feedback_loss = self.layer_feedback_loss(
                forward_layer=forward_layer_i,
                feedback_layer=feedback_layer_i,
                layer_input=layer_input,
                layer_output=layer_output,
            )
            layer_losses.append(layer_feedback_loss)
        return layer_losses

    def layer_feedback_loss(
        self,
        forward_layer: Callable[[Tensor], Tensor],
        feedback_layer: Callable[[Tensor], Tensor],
        layer_input: Tensor,
        layer_output: Tensor,
    ) -> Tensor:
        """Feedback loss calculation for a single forward and feedback layers."""
        x = layer_input.detach()
        y = layer_output.detach()

        # Reconstruct the layer input:
        x_hat = feedback_layer(layer_output.detach())

        if self.hp.n_noise_samples > 1:
            # If we use more noise samples then we "multiply" the batch size
            x_hat = tile_batch(x_hat, n=self.hp.n_noise_samples)
            x = tile_batch(x, self.hp.n_noise_samples)
            y = tile_batch(y, self.hp.n_noise_samples)

        # NOTE: A simple reconstruction loss would be just this:
        # reconstruction_error = F.mse_loss(x_hat, x)

        with torch.no_grad():
            dx = self.hp.feedback_noise_std * torch.randn_like(x)
            dy = self.hp.feedback_noise_std * torch.randn_like(y)

            noisy_x = x + dx
            noisy_y = y + dy

            output_noisy_x = forward_layer(noisy_x)

        # x + noise     ---forward layer--> y_hat_noisy_x
        # x_hat_noisy_x <--feedback layer-- y_hat_noisy_x
        x_hat_noisy_x = feedback_layer(output_noisy_x)
        dr_x = x_hat_noisy_x - x_hat

        # x_hat_noisy_y <--feedback-layer-- y + noise
        x_hat_noisy_y = feedback_layer(noisy_y)

        # Distance between `r` and the reconstructed `r` after y perturbation.
        dr_y = x_hat_noisy_y - x_hat

        dr_x_loss = -2 * (dx * dr_x).flatten(1).sum(1).mean()
        dr_y_loss = (dr_y**2).flatten(1).sum(1).mean()

        layer_loss = dr_x_loss + dr_y_loss
        return layer_loss

    def propagate_targets(
        self, x: Tensor, y: Tensor, forward_activations: list[Tensor]
    ) -> list[Tensor]:
        """Calculate the targets for each layer using the feedback network."""
        # Calculate the target for the last layer using the backward op of the F.cross_entropy fn:
        with torch.inference_mode(False), torch.set_grad_enabled(True):
            logits = forward_activations[-1]
            temp_logits = logits.detach().clone().requires_grad_(True)
            # NOTE: This is necessary when doing validation since torch.inference_mode was set
            # when this tensor was created.
            if y.is_inference():
                y = y.clone()
            ce_loss = F.cross_entropy(temp_logits, y, reduction="sum")
            logit_grads = torch.autograd.grad(
                ce_loss,
                temp_logits,
                only_inputs=True,  # Do not backpropagate further than the input tensor!
                create_graph=False,
            )
            assert len(logit_grads) == 1
            logit_grads = logit_grads[0]

            last_layer_delta = -self.hp.beta * logit_grads

        last_layer_target = logits.detach() + last_layer_delta

        # Targets for each forwrd layer. We construct this list by prepending to it.
        targets: list[Tensor] = [last_layer_target]
        target = last_layer_target
        with torch.no_grad():
            prev_target = target
            n_layers = len(self.feedback_network)
            for i, feedback_layer in zip(reversed(range(n_layers)), self.feedback_network):
                # NOTE: Normal "vanilla" target propagation would do this:
                # prev_target = feedback_layer(prev_target)

                # Our DTP method does this:
                prev_target = (
                    (forward_activations[i - 1] if i > 0 else x)
                    + feedback_layer(prev_target)
                    - feedback_layer(forward_activations[i])
                )
                targets.insert(0, prev_target)

            # # NOTE: Iterating through the feedback network like this starts from the output and
            # # moves back to the input.
            # for feedback_layer in self.feedback_network:
            #     # Create the target for layer[i-1] by applying the feedback layer to the target
            #     # of layer[i].
            #     target = feedback_layer(target)
            #     targets.insert(0, target)

        return targets

    def forward_loss(self, x: Tensor, y: Tensor, phase: PhaseStr) -> list[Tensor]:
        forward_activations = get_all_forward_activations(self.forward_network, x)
        targets = self.propagate_targets(x, y, forward_activations=forward_activations)

        # NOTE: How we have things set up, the feedback network also tries to reconstruct the
        # inputs. We don't really need to train that layer though, because it isn't useful for
        # training the forward net.
        assert targets[0].shape == x.shape
        target_activations = targets[1:]

        # The loss is the difference between the target activations and the actual activations.
        layer_losses = [
            (0.5 * ((activation_i - target_i) ** 2).view(activation_i.shape[0], -1).sum(1).mean())
            # F.mse_loss(activation_i, target_i)
            for activation_i, target_i in zip(forward_activations, target_activations)
        ]
        return layer_losses

    def _create_feedback_optimizer(self) -> torch.optim.Optimizer:
        feedback_optim_fn = instantiate(self.hp.feedback_optim)

        if isinstance(self.hp.feedback_optim.lr, (int, float)):
            return feedback_optim_fn(self.feedback_network.parameters())

        # Note: This block makes it possible for us to have a learning rate per layer.
        lrs_per_layer = list(self.hp.feedback_optim.lr)
        # We don't pass a learning rate for the first feedback layer.
        assert len(self.feedback_network) == len(lrs_per_layer) + 1
        # Any other parameters (e.g. first layer) will be optimized with this LR.
        default_lr = sum(lrs_per_layer) / len(lrs_per_layer)
        lrs_per_layer = [default_lr] + lrs_per_layer
        return feedback_optim_fn(
            [
                {"params": layer.parameters(), "lr": lr}
                for layer, lr in zip(self.feedback_network, lrs_per_layer)
            ],
            lr=default_lr,
        )

    def _create_forward_optimizer(self) -> torch.optim.Optimizer:
        forward_optim_fn = instantiate(self.hp.forward_optim)

        forward_optim = forward_optim_fn(self.forward_network.parameters())
        return forward_optim

    def configure_callbacks(self) -> list[Callback]:
        # NOTE: Can actually reuse this:
        # from beyond_backprop.algorithms.dtp.callbacks import CompareToBackpropCallback
        # from beyond_backprop.algorithms.example_target_prop.callbacks import
        # DetectIfTrainingCollapsed
        return super().configure_callbacks() + [
            # CompareToBackpropCallback(),
            # DetectIfTrainingCollapsed(),
        ]


def detached_forward(module: nn.Sequential, input: Tensor) -> Tensor:
    h = input
    for layer in module:
        h = layer(h.detach())
    return h
