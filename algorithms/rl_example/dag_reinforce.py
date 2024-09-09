from __future__ import annotations
from dataclasses import dataclass

import functools
from pathlib import Path
from typing import Any, Callable

import gym
import gym.spaces
import lightning
import numpy as np
import torch
from gym import spaces
from lightning import LightningModule
from torch import Tensor
from torch.distributions import Categorical, Normal

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo
from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.networks.fcnet import FcNet, fcnet_for_env
from beyond_backprop.utils.types import PhaseStr, StepOutputDict
from .rl_datamodule import RlDataModule, EpisodeBatch
from .feedback_alignment_utils import kolen_pollack_update, weight_mirror_update
from .utils import (
    check_and_normalize_box_actions,
)
from .dag_reinforce_layered_models import DAGRLModel
from .types import ActorOutput
from logging import getLogger as get_logger

logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")
eps = np.finfo(np.float32).eps.item()


class ExampleActorOutput(ActorOutput):
    """Additional outputs of the Actor (besides the action to take) for a single step in the env.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)

    In the case of Reinforce, we store the logits as well as the action log probabilities.
    """

    logits: Tensor
    """The network outputs at that step."""

    action_log_probability: Tensor
    """The log-probability of the selected action at that step."""


Episodes = EpisodeBatch[ExampleActorOutput]
"""The type of episodes that are received in the `training_step`, `validation_step`, etc. methods.

This just means "EpisodeBatch objects where the actor outputs are of type ExampleActorOutput".
"""


class DAGREINFORCE(Algorithm[FcNet, EpisodeBatch], LightningModule):
    """Example of a Reinforcement Learning algorithm: Reinforce.

    TODO: Figure out how to make this algorithm applicable in Supervised Learning as desired.
    """

    @dataclass
    class HParams(Algorithm.HParams):
        gamma: float = 0.99
        learning_rate: float = 1e-2
        weight_decay: float = 0.
        kolen_pollack: bool = False
        weight_mirror: bool = False
        kp_learning_rate: float = 0.
        kp_weight_decay: float = 0.
        wm_batch_size: int = 0

    def __init__(
        self,
        datamodule: RlDataModule[ExampleActorOutput],
        network: FcNet,
        hp: DAGREINFORCE.HParams | None = None,
    ):
        """
        Parameters
        ----------

        - env: Gym environment to train on.
        - gamma: Discount rate.
        """
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.network: FcNet
        self.hp: DAGREINFORCE.HParams
        self.datamodule: RlDataModule

        # if isinstance(datamodule.env.action_space, spaces.Box):

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate, weight_decay = self.hp.weight_decay)

    def forward(
        self,
        observations: Tensor,
        action_space: spaces.Discrete | spaces.Box,
    ) -> tuple[Tensor, ExampleActorOutput]:
        # NOTE: Would be nice to be able to do this:
        # assert observations.shape == self.network.input_space.shape
        # assert action_space.n == self.network.output_space.shape[0]
        actions = self.network(observations).view([])
        network_outputs = self.network.ts[-1].dist_params
        action_log_probabilities = torch.stack(list(node.log_prob() for node in self.network.graph.nodes)).sum(dim = 0)
        actor_outputs_to_save_in_episode: ExampleActorOutput = {
            "logits": network_outputs,
            "action_log_probability": action_log_probabilities,
        }
        return actions, actor_outputs_to_save_in_episode

    def training_step(self, batch: EpisodeBatch[ExampleActorOutput]) -> StepOutputDict:
        return self.shared_step(batch, phase="train")

    # NOTE: For some reason PL requires us to have a second positional argument for the batch_index
    # even if it isn't used, but the training step doesn't need it.
    def validation_step(
        self, batch: EpisodeBatch[ExampleActorOutput], batch_index: int
    ) -> StepOutputDict:
        return self.shared_step(batch, phase="val")

    def shared_step(
        self, batch: EpisodeBatch[ExampleActorOutput], phase: PhaseStr
    ) -> StepOutputDict:
        """Perform a single step of training or validation.

        The input is a batch of episodes, and the output is a dictionary with the loss and metrics.
        PyTorch-Lightning will then use the loss as the training signal, but we could also do the
        backward pass ourselves if we wanted to (as shown in the ManualGradientsExample).
        """
        rewards = batch["rewards"]
        batch_size = rewards.size(0)

        # Nested Tensor of shape [n_envs, <episode_len>] where episode_len varies between tensors.
        returns = discounted_returns(rewards, gamma=self.hp.gamma)

        # NOTE: Equivalent to the following:
        # normalized_returns = (returns - returns.mean(dim=1)) / (returns.std(dim=1) + eps)
        normalized_returns = torch.nested.as_nested_tensor(
            [(ret - ret.mean()) / (ret.std() + eps) for ret in returns.unbind()]
        )
        # Retrieve the outputs that we saved at each step:
        actor_outputs = batch["actor_outputs"]

        # NOTE: In this particular case here, the actions are "live" tensors with grad_fns.
        # For Off-policy-style algorithms like DQN, this could be sampled from a replay buffer, and
        # so we could pass all the episodes through the network in a single forward pass (thanks to
        # the nested tensors).

        # Nested tensor of shape [n_envs, <episode_len>]
        action_log_probs = actor_outputs["action_log_probability"].reshape_as(normalized_returns)
        policy_loss_per_step = -action_log_probs * normalized_returns

        # Sum across episode steps
        policy_loss_per_episode = policy_loss_per_step.to_padded_tensor(0.0).sum(dim=1)
        # Average across episodes
        policy_loss = policy_loss_per_episode.mean(dim=0)
        self.log(f"{phase}/loss", policy_loss, prog_bar=True)

        if self.hp.kolen_pollack:
            kolen_pollack_update(self.network, self.hp)
        elif self.hp.weight_mirror:
            weight_mirror_update(self.network, self.hp)
        # Log the episode statistics gathered by the RecordEpisodeStatistics gym wrapper.
        episode_stats = [
            episode_infos[-1]["episode"]
            for episode_infos in batch["infos"]
            if episode_infos and "episode" in episode_infos[-1]
        ]
        if episode_stats:
            episode_lengths = np.array([s["l"] for s in episode_stats])
            episode_total_rewards = np.array([s["r"] for s in episode_stats])
            # episode_time_since_start = np.array([s["t"] for s in episode_stats])  # unused atm.
            avg_episode_length = sum(episode_lengths) / batch_size
            avg_episode_reward = episode_total_rewards.mean(0)
            avg_episode_return = sum(returns.select(dim=1, index=0)) / batch_size
            log_kwargs = dict(prog_bar=True, batch_size=batch_size)
            self.log(f"{phase}/avg_episode_length", avg_episode_length, **log_kwargs)
            self.log(f"{phase}/avg_episode_reward", avg_episode_reward, **log_kwargs)
            self.log(f"{phase}/avg_episode_return", avg_episode_return, **log_kwargs)

        return {"loss": policy_loss}

    def on_fit_start(self) -> None:
        logger.info("Starting training.")
        assert isinstance(self.datamodule, RlDataModule) or hasattr(self.datamodule, "set_actor")
        # Set the actor on the datamodule so our `forward` method is used to select actions at each
        # step.
        self.datamodule.set_actor(self)

        # We only add the gym wrappers to the datamodule once.
        assert self.datamodule.train_dataset is None
        assert (
            len(self.datamodule.train_wrappers) == 0
            or self.datamodule.train_wrappers[-1] is not RecordEpisodeStatistics
        )
        self.datamodule.train_wrappers += self.gym_wrappers_to_add(videos_subdir="train")
        self.datamodule.valid_wrappers += self.gym_wrappers_to_add(videos_subdir="valid")
        self.datamodule.test_wrappers += self.gym_wrappers_to_add(videos_subdir="test")

    def gym_wrappers_to_add(self, videos_subdir: str) -> list[Callable[[gym.Env], gym.Env]]:
        return [
            check_and_normalize_box_actions,
            # NOTE: The functools.partial below is Equivalent to the following:
            # lambda env: RecordVideo(env, video_folder=str(log_dir / "videos/train")),
            functools.partial(
                RecordVideo, video_folder=str(self.log_dir / "videos" / videos_subdir)
            ),
            RecordEpisodeStatistics,
        ]

    @property
    def log_dir(self) -> Path:
        """Returns  the Trainer's log dir if we have a trainer.

        (NOTE: we should always have one, except maybe during some unit tests where the DataModule
        is used by itself.)
        """
        log_dir = Path("logs/default")
        if self.trainer is not None:
            log_dir = Path(self.trainer.log_dir or log_dir)
        return log_dir


# def get_action_distribution(
#     network_outputs: Tensor,
#     action_space: spaces.Discrete | spaces.Box,
# ) -> Categorical | Normal:
#     """Creates an action distribution for the given action space based on these network outputs."""
#     if isinstance(action_space, spaces.Discrete):
#         return Categorical(logits=network_outputs)

#     # NOTE: The environment has a wrapper applied to it that normalizes the continuous action space
#     # to be in the [-1, 1] range, and the actions outside that range will be clipped by that
#     # wrapper.
#     assert isinstance(action_space, spaces.Box)
#     assert (action_space.low == -1).all() and (action_space.high == 1).all()
#     d = action_space.shape[-1]
#     assert network_outputs.shape[-1] == d * 2

#     loc, scale = network_outputs.chunk(2, -1)
#     loc = torch.tanh(loc)
#     scale = torch.relu(scale) + 1e-5
#     return Normal(loc=loc, scale=scale)


def discounted_returns(rewards_batch: Tensor, gamma: float) -> Tensor:
    """Returns a batch of discounted returns for each step of each episode.

    Parameters
    ----------
    rewards_batch: A (possibly nested) tensor of shape [b, `ep_len`] where `ep_len` may vary.
    gamma: The discount factor.

    Returns
    -------
    A (possibly nested) tensor of shape [b, `ep_len`] where `ep_len` may vary.
    """
    # todo: Check if this also works if the rewards batch is a regular tensor (with all the
    # episodes having the same length).
    # _batch_size, ep_len = rewards.shape

    # NOTE: `rewards` has shape [batch_size, <ep_length>] atm.
    assert rewards_batch.is_nested
    returns_batch: list[Tensor] = []

    for rewards in rewards_batch.unbind():
        returns = torch.zeros_like(rewards)

        discounted_future_rewards = torch.zeros_like(rewards[0])

        ep_len = rewards.size(0)

        for step in reversed(range(ep_len)):
            reward_at_that_step = rewards[step]
            discounted_future_rewards = reward_at_that_step + gamma * discounted_future_rewards
            returns[step] = discounted_future_rewards

        returns_batch.append(returns)

    return torch.nested.as_nested_tensor(returns_batch)


# def _discounted_returns_list(rewards: list[float], gamma: float) -> list[float]:
#     sum_of_discounted_future_rewards = 0
#     returns_list: deque[float] = deque()
#     for reward in reversed(rewards):
#         sum_of_discounted_future_rewards = reward + gamma * sum_of_discounted_future_rewards
#         returns_list.appendleft(sum_of_discounted_future_rewards)  # type: ignore
#     return list(returns_list)


def main():
    env_fn = functools.partial(gym.make, "CartPole-v1", render_mode="rgb_array")
    env: gym.Env = env_fn()
    datamodule = RlDataModule(env=env_fn, actor=None, episodes_per_epoch=100, batch_size=10)

    # TODO: Test out if we can make this stuff work with Brax envs:
    # from brax import envs
    # import brax.envs.wrappers
    # from brax.envs import create
    # from brax.envs import wrappers
    # from brax.io import metrics
    # from brax.training.agents.ppo import train as ppo
    # env = create("halfcheetah", batch_size=2, episode_length=200, backend="spring")
    # env = wrappers.VectorGymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    # env = wrappers.TorchWrapper(env, device=torch.device("cuda"))

    network = DAGRLModel(env.observation_space, env.action_space)

    algorithm = DAGREINFORCE(datamodule=datamodule, network=network)

    datamodule.set_actor(algorithm)

    trainer = lightning.Trainer(max_epochs=2, devices=1, accelerator="auto")
    # todo: fine for now, but perhaps the SL->RL wrapper for Reinforce will change that.
    assert algorithm.datamodule is datamodule
    trainer.fit(algorithm, datamodule=datamodule)

    # Otherwise, could also do it manually, like so:

    # optim = algorithm.configure_optimizers()
    # for episode in algorithm.train_dataloader():
    #     optim.zero_grad()
    #     loss = algorithm.training_step(episode)
    #     loss.backward()
    #     optim.step()
    #     print(f"Loss: {loss}")


if __name__ == "__main__":
    main()
