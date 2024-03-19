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
import torch
import gymnasium as gym
import time
import argparse
from torch.distributions import Categorical

from qrl_utils import QRLCartPoleWrapper, Buffer, policy_gradient, validate, parse
from circuit import generate_circuit

# for qiskit-torch-module
from qiskit_torch_module import QuantumModule

# for qiskit-machine-learning
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# This is only for my machine, suppresses some CUDA warnings  # TODO remove
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def quantum_policy_gradients(
        env_name: str,
        vqc_depth: int = 1,
        episodes: int = 1000,
        batch_size: int = 10,
        gamma: float = 0.99,
        val_freq: int = None,
        val_runs: int = 100,
        early_stopping: float = None,
        seed: int = None,
        use_qml: bool = False
):
    """ Realizes the quantum policy gradient algorithm described in
        N. Meyer et al., Quantum Policy Gradient Algorithm with Optimized Action Decoding, PMLR 202:14592-24613 (2023).
        using the qiskit-torch-module

        Args:
            env_name: Which environment to train on, either `CartPole-v0` or `CartPole-v1`
            vqc_depth: Depth of the policy-approximation quantum circuit
            episodes: Number of episodes to train for (number of epochs will be episodes//batch_size)
            batch_size: Number of simultaneous trajectories to consider in each epoch
            gamma: Discount factor
            val_freq: Frequency (w.r.t. epochs) of validation, by default de-activated
            val_runs: Number of validation runs
            early_stopping: Threshold for early stopping, by default de-activated
            seed: Allows for reproducible runs
            use_qml: Use qiskit-machine-learning instead of qiskit-torch-module
    """

    if env_name not in ['CartPole-v0', 'CartPole-v1']:
        raise ValueError('The environment {} is currently not implemented. Use `CartPole-v0/1`.'.format(env_name))
    if val_freq is None and early_stopping is not None:
        raise ValueError('Early stopping can only be used if validation is activated.')

    # initialize environments for training and validation
    env = gym.make_vec(env_name, num_envs=batch_size, vectorization_mode='sync', wrappers=[QRLCartPoleWrapper])
    env_val = gym.make_vec(env_name, num_envs=val_runs, vectorization_mode='sync', wrappers=[QRLCartPoleWrapper])

    # data buffers for trajectories
    buffer = Buffer(batch_size, gamma=gamma)
    buffer_val = Buffer(val_runs)

    # generate VQC used for policy approximation
    vqc, params_encoding, (params_variational, params_scaling) = generate_circuit(num_qubits=env.observation_space.shape[1],
                                                                                  depth=vqc_depth,
                                                                                  entanglement_structure='full',
                                                                                  input_scaling=True)

    # quantum neural network that handles gradient computation via PyTorch
    # initialize variational parameters with Normal(mean=0.0, std=0.1); initialize scaling parameters to 1.0
    if use_qml:
        # for qiskit-machine-learning it is necessary to first define a quantum neural network, and subsequently
        # connect it to pytorch.
        # it is not possible to define multiple parameter sets, so variational and scaling parameters have to be combined.
        qnn = EstimatorQNN(
            circuit=vqc, input_params=params_encoding.params,
            weight_params=params_variational.params + params_scaling.params,
            observables=[SparsePauliOp('ZZZZ')], gradient=ReverseEstimatorGradient()
        )
        initial_weights = (list(0.1 * np.random.randn(len(params_variational.params)))
                           + list(np.ones(len(params_scaling.params))))
        model = TorchConnector(qnn, initial_weights=initial_weights)
        # set up optimizer, as we have only one parameter set handle a combined learning rate has to be used
        opt = torch.optim.Adam(model.parameters(), lr=0.05, amsgrad=True)  # noqa
    else:
        # directly set up quantum model
        model = QuantumModule(vqc, encoding_params=params_encoding, variational_params=[params_variational, params_scaling],
                              variational_params_names=['variational', 'scaling'],
                              variational_params_initial=[('normal', {'std': 0.1}), ('constant', {'val': 1.0})],
                              observables='tensoredZ', num_threads_forward=0, num_threads_backward=0, seed_init=seed)
        # set individual learning rates for the different parameter sets
        opt = torch.optim.Adam([{'params': model.variational, 'lr': 0.01}, {'params': model.scaling, 'lr': 0.1}], amsgrad=True)

    epochs = episodes // batch_size
    print('Training for {} Epochs with a batch size of {}.\n'.format(epochs, batch_size))

    start_time, time_validation = time.perf_counter(), 0.0
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    for epoch in range(epochs):

        # handle validation and early stopping
        if val_freq is not None:
            if 0 == epoch % val_freq:
                start_time_validation = time.perf_counter()
                val_reward, early_stop = validate(env_val, buffer_val, model, early_stopping_criterion=early_stopping)
                time_validation += time.perf_counter() - start_time_validation
                print('VALIDATE [Epoch {}] Avg. Reward: {:.3f}     (max: {:.1f}, min: {:.1f})'.format(epoch, *val_reward))
                if early_stop:
                    print('\nEarly stopping, the desired reward value was achieved!')
                    break

        # reset data buffer
        buffer.reset()

        # guarantees deterministic behavior if seed is set and random behavior if not
        observations, _ = env.reset(seed=[np.random.randint(1000) for _ in range(batch_size)])

        if not use_qml:
            # compute forward passes for data generation sequentially, as batch_size typically to low as that
            # parallelization brings an advantage (for qiskit-machine-learning this has no effect)
            num_threads_forward = model.num_threads_forward
            model.set_num_threads_forward(1)

        # collect trajectories until all environments have terminated
        while True:
            # store observations
            buffer.append_observations(observations)

            # we do not record gradients as of now, this allows for a much faster batch-parallel backward pass later on
            with torch.set_grad_enabled(False):
                # sample action following current policy -- for entire batch at once
                # policy: [pi(a=0|obs), pi(a=1|obs)] = [(model(obs) + 1) / 2, (-model(obs) + 1) / 2]
                policies = torch.div(torch.add(torch.mul(model(torch.tensor(observations)).repeat(1, 2),
                                                         torch.tensor([1.0, -1.0])), 1.0), 2.0)
                # sample action from distribution defined by policy
                dist = Categorical(policies)
                actions = [action.item() for action in dist.sample()]

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

        if not use_qml:
            # set back to parallel computation for computing gradients (for qiskit-machine-learning this has no effect)
            model.set_num_threads_forward(num_threads_forward)  # noqa

        # print some statistics
        print('TRAIN [Epoch {}->{}, Episodes {}-{}, Total Steps: {}]  Avg. Reward: {:.3f}     (max: {:.1f}, min: {:.1f})'
              .format(epoch, epoch+1, epoch*batch_size+1, (epoch+1) * batch_size, buffer.get_total_steps, *buffer.get_epoch_reward))

        # batched forward pass
        # policy: [pi(a=0|obs), pi(a=1|obs)] = [(model(obs) + 1) / 2, (-model(obs) + 1) / 2]
        batched_policies = torch.div(torch.add(torch.mul(model(torch.tensor(buffer.batched_observations)).repeat(1, 2),
                                                         torch.tensor([1.0, -1.0])), 1.0), 2.0)
        # filter by executed action, compute log
        batched_log_policies = torch.log(torch.gather(batched_policies, 1,
                                                      torch.LongTensor(buffer.batched_actions).unsqueeze(1)))

        # optimize parameters with collected trajectories
        opt.zero_grad()
        loss = policy_gradient(batched_log_policies, buffer.batched_rewards)
        loss.backward()
        opt.step()

    time_total = time.perf_counter() - start_time
    print('\nTotal time: {:.1f}s     (thereof {:.1f}s for validation)'.format(time_total, time_validation))


if __name__ == '__main__':
    _args = parse()
    quantum_policy_gradients(
        _args.environment,
        vqc_depth=_args.depth,
        episodes=_args.episodes,
        batch_size=_args.batch,
        gamma=0.99,
        val_freq=5,
        val_runs=100,
        early_stopping=195.0 if 'CartPole-v0' == _args.environment else 475.0,  # default values from gymnasium
        seed=_args.seed,
        use_qml=_args.use_qml
    )
