# This code is part of the Qiskit-Torch-Module.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import gymnasium as gym
import operator
import torch
from torch.distributions import Categorical
import argparse

from qiskit_torch_module import QuantumModule
from qiskit_machine_learning.connectors import TorchConnector


class QRLCartPoleWrapper(gym.Wrapper):
    """
    Wrapper for the CartPole environment
    """

    def __init__(self, env):
        """
        Environment wrapper

            Args:
                env: original environment handle.
        """
        super().__init__(env)
        self.env = env

    def step(self, action):
        """ Execute one step in the environment, normalize observations to be in [-pi, pi]. """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[0] /= 2.4
        obs[1] /= 2.5
        obs[2] /= 0.21
        obs[3] /= 2.5
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """ Reset environment. """
        return self.env.reset(**kwargs)


class Buffer:
    """
    Trajectory data holder.
    """

    def __init__(self, batch_size, gamma=1.0):
        """
        Trajectory data.

            Args:
                batch_size: Number of simultaneously tracked trajectories.
                gamma: Discount factor.
        """
        assert 0 <= gamma <= 1, 'The discount factor `gamma` has t be between in [0.0, 1.0].'
        self.batch_size = batch_size
        self.gamma = gamma
        self.total_steps = 0
        self.running_rewards = {}
        self.epoch_counter = 0
        self.epoch_rewards = None
        self.terminated, self.truncated = None, None
        self.observations, self.actions, self.rewards = None, None, None

    def reset(self):
        """ Empty data buffers, reset termination flags. """
        self.epoch_rewards = None
        self.terminated = np.array([False for _ in range(self.batch_size)])
        self.truncated = np.array([False for _ in range(self.batch_size)])
        self.observations = [[] for _ in range(self.batch_size)]
        self.actions = [[] for _ in range(self.batch_size)]
        self.rewards = [[] for _ in range(self.batch_size)]

    def append_observations(self, observations):
        """ Append observations to buffer. """
        assert self.observations is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, observation in zip(self._not_done, observations):
            self.observations[batch_index].append(observation)

    def append_actions(self, actions):
        """ Append actions to buffer. """
        assert self.actions is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, action in zip(self._not_done, actions):
            self.actions[batch_index].append(action)
        # construct 0-padded actions to account for already terminated environments
        actions_padded = np.zeros((self.batch_size,), dtype=int)
        actions_padded[self._not_done] = actions
        return actions_padded

    def append_rewards(self, rewards):
        """ Append rewards to buffer. """
        assert self.rewards is not None, 'Forgot to call `reset(batch_size)`.'
        for batch_index, reward in zip(self._not_done, rewards):
            self.rewards[batch_index].append(reward)

    def update_terminated_and_truncated(self, terminated, truncated):
        """ Update termination flags. """
        # select those that were not terminated/truncated in the previous step
        indices_not_done = self._not_done
        terminated, truncated = terminated[indices_not_done], truncated[indices_not_done]
        self.terminated[indices_not_done], self.truncated[indices_not_done] = terminated, truncated
        # return the indices of the environments that are still not done (i.e. terminated or truncated)
        return self._not_done

    def finish_epoch(self):
        """ Terminate epoch, potentially compute discounted rewards. """
        assert self.observations is not None and self.actions is not None and self.rewards is not None, 'Forgot to call `reset(batch_size)`.'
        self.observations = self._flatten_list(self.observations)
        self.actions = self._flatten_list(self.actions)

        # store for logging
        self.epoch_rewards = [sum(rewards) for rewards in self.rewards]
        self.running_rewards[self.epoch_counter] = np.average(self.epoch_rewards)
        # discount rewards
        self.rewards = self._flatten_list([self._discount_and_cumulate_rewards(rewards) for rewards in self.rewards])

        assert len(self.observations) == len(self.actions) == len(self.rewards), 'Inconsistent trajectory.'
        self.total_steps += len(self.observations)
        self.epoch_counter += 1

    @property
    def batched_observations(self):
        """ Return stored observations. """
        assert self.observations is not None, 'Forgot to call `finish_epoch()`.'
        return self.observations

    @property
    def batched_actions(self):
        """ Return stores actions. """
        assert self.actions is not None, 'Forgot to call `finish_epoch()`.'
        return self.actions

    @property
    def batched_rewards(self):
        """ Return stores rewards. """
        assert self.rewards is not None, 'Forgot to call `finish_epoch()`.'
        return self.rewards

    @property
    def get_epoch_reward(self):
        """ Return average reward of this epoch. """
        assert self.epoch_rewards is not None, 'Forgot to call `finish_epoch()`.'
        return np.average(self.epoch_rewards), np.max(self.epoch_rewards), np.min(self.epoch_rewards)

    @property
    def get_total_steps(self):
        """ Return total number of steps since buffer initialization. """
        return self.total_steps

    @property
    def get_running_rewards(self):
        """ Return cumulated rewards of this epoch. """
        return self.running_rewards

    @property
    def _not_done(self):
        """ Return indices of environments thar are not done (terminated or truncated)
        """
        assert self.terminated is not None and self.truncated is not None, 'Forgot to call `reset(batch_size)`.'
        return np.where(list(map(operator.not_, map(operator.or_, self.terminated, self.truncated))))[0]

    @staticmethod
    def _flatten_list(nested_list):
        """ Helper function that flattens a list of lists. """
        flat_list = []
        for individual_list in nested_list:
            flat_list += individual_list
        return flat_list

    def _discount_and_cumulate_rewards(self, rewards):
        """ Cumulate (and potentially discount) rewards of this epoch. """
        discounted_cumulated_rewards = []
        running_cumulated_reward = 0
        for t in range(len(rewards) - 1, -1, -1):
            running_cumulated_reward = rewards[t] + self.gamma * running_cumulated_reward
            discounted_cumulated_rewards.append(running_cumulated_reward)
        discounted_rewards = discounted_cumulated_rewards[::-1]
        return discounted_rewards


def policy_gradient(log_policies, rewards):
    """
    Compute policy gradient, i.e. avg[ \nabla ln pi (A_i | S_i) G_i ]. We need to use the negative value, as
    torch.optim by default does gradient descend (but policy gradients algorithms require gradient ascend)

        Args:
            log_policies: Log-values of policy.
            rewards: Associated (discounted) rewards.
        Return:
            Policy gradient loss.
    """
    return torch.neg(torch.mean(torch.mul(log_policies.squeeze(1), torch.tensor(rewards))))


def validate(
        env: gym.vector.VectorEnv,
        buffer: Buffer,
        model: QuantumModule | TorchConnector,
        early_stopping_criterion: float = None
) -> tuple[tuple[float, float, float], bool]:
    """
    Validate given model.

        Args:
            env: Environment handle.
            buffer: Data buffer for tracking results.
            model: Model to evaluate.
            early_stopping_criterion: Evaluation criterion for early stopping.
        Return:
            Validation results, early stopping flag
    """
    # reset data buffer (no need to store observations, as we do not perform updates here)
    buffer.reset()

    # guarantees deterministic behavior if seed is set and random behavior if not
    observations, _ = env.reset(seed=[np.random.randint(1000) for _ in range(env.num_envs)])

    while True:

        # store observations (only for consistency, are never used for validation)
        buffer.append_observations(observations)

        # no need to track gradients for validation
        with torch.set_grad_enabled(False):
            # sample action following current policy -- for entire batch at once
            # policy: [pi(a=0|obs), pi(a=1|obs)] = [(model(obs) + 1) / 2, (-model(obs) + 1) / 2]
            policies = torch.div(torch.add(torch.mul(model(torch.tensor(observations)).repeat(1, 2),
                                                     torch.tensor([1.0, -1.0])), 1.0), 2.0)
            # sample action from distribution defined by policy
            dist = Categorical(policies)
            actions = dist.sample()

        # store actions, returns 0-padded actions (as some environments might have already terminated)
        # (this is a small workaround, as gym.VectorEnv automatically resets terminated environments)
        actions_padded = buffer.append_actions(actions)

        # execute actions in respective environments
        observations_, rewards, terminated, truncated, _ = env.step(actions_padded)

        # store rewards
        buffer.append_rewards(rewards)

        # update termination and truncation flags
        # returns indices of environments that are still not done -> use to select only relevant observations
        indices_not_done = buffer.update_terminated_and_truncated(terminated, truncated)
        observations = observations_[indices_not_done]

        # check if all environments are done
        if 0 == len(indices_not_done):
            buffer.finish_epoch()
            break

    # evaluate results, optionally evaluate early-stopping criterion
    val_reward = buffer.get_epoch_reward
    early_stop = False
    if early_stopping_criterion is not None:
        # test for average reward
        if val_reward[0] >= early_stopping_criterion:
            early_stop = True

    return val_reward, early_stop


def parse():
    parser = argparse.ArgumentParser()
    choices_env = ['CartPole-v0', 'CartPole-v1']
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth of used VQC.')
    parser.add_argument('--batch', type=int, default=10,
                        help='Number of trajectories to sample for each update step.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Maximum number of episodes to train for.')
    parser.add_argument('--environment', '-env', type=str, default='CartPole-v1', choices=choices_env,
                        help='Which environment to train on.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set seed for reproducibility.')
    parser.add_argument('--use_qml', action='store_true',
                        help='Train using qiskit-machine-learning instead of qiskit-torch-module.')
    return parser.parse_args()
