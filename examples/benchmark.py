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
import argparse
import time
import torch
import tqdm
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ParamShiftEstimatorGradient, SPSAEstimatorGradient

# imports from qiskit-machine-learning for benchmark baselines
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# import proposed module
from qiskit_torch_module import QuantumModule

# generate circuits
from circuit import generate_circuit


def benchmark(
        qubits: int = 12,
        depth: int = 3,
        observables: int = 0,
        batch_size: int = 128,
        repeat_experiment: int = 10,
        num_threads: int = 0,
        grad_method: str = 'reverse',
        use_qml: bool = False
):
    """ Run timed forward and backward passed for either qiskit-machine-learning or qiskit-torch-module

        Args:
            qubits: Number of qubits in quantum circuits
            depth: Number of layers in quantum circuit
            observables: Number of observables to evaluate for. By default, single-qubit Pauli-Z observables are measured
                on all available qubits
            batch_size: Batch of input data to benchmark on
            repeat_experiment: Runs the experiment multiple times (with different initializations) and averages the runtimes
            num_threads: Number of threads to use for qiskit-torch-module. All available by default
            grad_method: Gradient computation method to use, either `reverse`, `ps` for parameter-shift, `spsa`
            use_qml: Use qiskit-machine-learning instead of qiskit-torch-module
    """

    assert grad_method in ['reverse', 'ps', 'spsa'], ('Gradient computation method `{}` is not known. Choose either '
                                                      '`reverse`, `ps`, or `spsa`'.format(grad_method))

    print('Qubits: {}, Depth: {}, Observables: {}, Batch size: {}, Framework: {}, {}Gradient computation method: {}, '
          'Repetitions: {}'.format(qubits, depth, observables, batch_size,
                                   'qiskit-machine-learning' if use_qml else 'qiskit-torch-module',
                                   '' if use_qml else 'Number of threads: {}, '.format('all' if 0 == num_threads else num_threads),
                                   grad_method, repeat_experiment))

    # set up VQC
    vqc, encoding, (variational, scaling) = generate_circuit(num_qubits=qubits, depth=depth, input_scaling=True)

    # concatenate both trainable parameter sets
    # (on the contrary to qiskit-machine-learning, it is possible to just provided different trainable parameter sets to
    # qiskit-torch-module, but for a fair comparison the combined parameter set is used in both cases)
    weights = variational.params + scaling.params
    print('Number of variational parameters: {}'.format(len(weights)))

    if use_qml:
        # set up quantum model with qiskit-machine-learning
        match grad_method:
            case 'reverse':
                grad_fn = ReverseEstimatorGradient()
            case 'ps':
                grad_fn = ParamShiftEstimatorGradient(Estimator())
            case 'spsa':
                # batch size is the number of SPSA-approximations to compute, here 1 by default
                grad_fn = SPSAEstimatorGradient(Estimator(), epsilon=0.1, batch_size=1)
            case _:
                raise ValueError(f'Gradient method {grad_method} unknown for qiskit-machine-learning.')
        qnn = EstimatorQNN(
            circuit=vqc, input_params=encoding.params, weight_params=weights,
            # if qubits==observables single-qubit PauliZ observables on all qubits, same as done for
            # `observables='individualZ'` in QTM, otherwise also `observables` many tensored Pauli-Z observables
            observables=[SparsePauliOp(i * 'I' + 'Z' + (qubits - i - 1) * 'I') for i in range(qubits)]
            if qubits == observables else [SparsePauliOp(qubits * 'Z') for _ in range(observables)],
            gradient=grad_fn
        )
        model = TorchConnector(qnn)
    else:
        # set up quantum model with qiskit-torch-module
        if grad_method != 'reverse':
            raise ValueError(f'Gradient method {grad_method} unknown for qiskit-torch-module.')
        model = QuantumModule(
            vqc, encoding_params=encoding, variational_params=weights,
            variational_params_initial=('uniform', {'a': -1.0, 'b': 1.0}),  # for consistency with qml
            # in case qubits==observables measure single-qubit Pauli observables on all qubits, otherwise just
            # measure `observable` many tensored Pauli-Z observables. Obviously, this would not be done in practice, but
            # for time benchmarking the actual observables are of not too much relevance
            observables='individualZ'
            if qubits == observables else [SparsePauliOp(qubits*'Z') for _ in range(observables)],
            num_threads_forward=num_threads, num_threads_backward=num_threads
        )

    # evaluate timings over several independent runs to get better accuracy
    times_forward, times_backward, times_combined = [], [], []
    for _ in tqdm.tqdm(range(repeat_experiment)):

        # draw random input data of size `batch_size`
        data = torch.rand((batch_size, qubits))

        # timed forward pass
        start_time_forward = time.perf_counter()
        data = model(data)
        times_forward.append(time.perf_counter() - start_time_forward)

        # just compute a very simple loss term (batch-mean of sum of observed expectation values)
        loss = torch.mean(torch.sum(data, dim=1))

        # timed backward pass
        start_time_backward = time.perf_counter()
        loss.backward()
        times_backward.append(time.perf_counter() - start_time_backward)

        times_combined.append(times_forward[-1] + times_backward[-1])

    print('Combined:   {:.3f}s     (max: {:.3f}s, min: {:.3f}s)'
          .format(np.average(times_combined), np.max(times_combined), np.min(times_combined)))
    print('Forward:    {:.3f}s     (max: {:.3f}s, min: {:.3f}s)'
          .format(np.average(times_forward), np.max(times_forward), np.min(times_forward)))
    print('Backward:   {:.3f}s     (max: {:.3f}s, min: {:.3f}s)'
          .format(np.average(times_backward), np.max(times_backward), np.min(times_backward)))


def parse():
    parser = argparse.ArgumentParser()
    choices_grad = ['reverse', 'ps', 'spsa']
    parser.add_argument('--qubits', type=int, default=12,
                        help='Width of VQC.')
    parser.add_argument('--depth', type=int, default=3,
                        help='Depth of VQC.')
    parser.add_argument('--observables', type=int, default=0,
                        help='Number of observables to evaluate (by default equal no number of qubits).')
    parser.add_argument('--batch', type=int, default=48,
                        help='Batch size of input data.')
    parser.add_argument('--repeat', type=int, default=10,
                        help='Number of runs to average over.')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of parallel workers to use in qtm (by default all available).')
    parser.add_argument('--grad', type=str, default='reverse', choices=choices_grad,
                        help='Gradient computation method.')
    parser.add_argument('--use_qml', action='store_true',
                        help='Train using qiskit-machine-learning instead of qiskit-torch-module.')
    args = parser.parse_args()
    if 0 == args.observables:
        args.observables = args.qubits
    return args


if __name__ == '__main__':
    _args = parse()
    benchmark(
        qubits=_args.qubits,
        depth=_args.depth,
        observables=_args.observables,
        batch_size=_args.batch,
        repeat_experiment=_args.repeat,
        num_threads=_args.threads,
        grad_method=_args.grad,
        use_qml=_args.use_qml
    )
