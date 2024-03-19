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
import time
import os

from circuit import generate_circuit
from qml_utils import parse, validate
from torchvision import datasets
from torchvision import transforms as transforms
from torch.utils.data import DataLoader

# for qiskit-torch-module
from qiskit_torch_module import QuantumModule

# for qiskit-machine-learning
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def quantum_classification(
        episodes: int = 1000,
        batch_size: int = 48,
        val_freq: int = None,
        seed: int = None,
        use_qml: bool = False
):
    """ Realizes the quantum classification algorithm described in
        M. Periyasamy et al., Incremental Data-Uploading for Full-Quantum Classification, IEEE QCE 1:31-37 (2022).
        using the qiskit-torch-module

        Args:
            episodes: Number of episodes to train for (number of epochs will be episodes//batch_size)
            batch_size: Number of simultaneous samples to consider in each epoch
            val_freq: Frequency (w.r.t. epochs) of validation, by default de-activated
            seed: Allows for reproducible runs
            use_qml: Use qiskit-machine-learning instead of qiskit-torch-module
    """

    # re-scales MNIST images from 28x28 to 10x10, load training and test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((10, 10))])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # for training batches of size `batch_size`, for validation 1000 random samples (i.e. 10% of test dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

    # generate VQC used for function approximation
    vqc, params_encoding, (params_variational, _) = generate_circuit(num_qubits=10, depth=10,
                                                                     incremental_data_uploading=True)

    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    if use_qml:
        # for qiskit-machine-learning it is necessary to first define a quantum neural network, and subsequently
        # connect it to pytorch.
        qnn = EstimatorQNN(
            circuit=vqc,
            input_params=params_encoding.params,
            weight_params=params_variational.params,
            # individual Pauli-Z measurements
            observables=[SparsePauliOp(index * 'I' + 'Z' + (10 - index - 1) * 'I') for index in range(10)],
            gradient=ReverseEstimatorGradient()
        )
        model = TorchConnector(qnn)
    else:
        # directly set up quantum model
        model = QuantumModule(
            vqc,
            encoding_params=params_encoding,
            variational_params=params_variational,
            observables='individualZ',
            num_threads_forward=0, num_threads_backward=0)

    # set up optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = episodes // batch_size
    print('Training for {} Epochs with a batch size of {}.\n'.format(epochs, batch_size))

    start_time, time_validation = time.perf_counter(), 0.0

    for epoch, (data, target) in enumerate(train_dataloader):
        if epoch >= epochs:
            break

        # handle validation
        if val_freq is not None and 0 == epoch % val_freq:
            time_validation += validate(model, val_dataloader, epoch=epoch)

        # flatten batched data for encoding
        data = torch.reshape(data, (data.shape[0], -1))
        opt.zero_grad()
        # compute forward pass and apply softmax
        prediction = torch.nn.functional.softmax(model(data), dim=1)
        # compute cross-entropy loss
        loss = loss_fn(prediction, target)
        print('TRAIN [Epoch {}->{}, Samples {}-{}]  Loss: {:.5f}, Correct {}/{}'
              .format(epoch, epoch+1, epoch*batch_size+1, (epoch+1) * batch_size, loss.item(),
                      torch.sum(torch.eq(torch.argmax(prediction, dim=1), target)), batch_size))
        # compute gradients
        loss.backward()
        # perform update
        opt.step()

    total_time = time.perf_counter() - start_time
    print('\nTotal time: {:.1f}s     (thereof {:.1f}s for validation)'
          .format(total_time, time_validation))


if __name__ == '__main__':
    _args = parse()
    quantum_classification(
        episodes=_args.episodes,
        batch_size=_args.batch,
        val_freq=10,  # validate all 10 epochs
        seed=_args.seed,
        use_qml=_args.use_qml
    )
