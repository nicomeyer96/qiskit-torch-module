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
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _equal_superposition(vqc, num_qubits):
    """ Creates initial equal superposition over all computational basis states
    """
    for q in range(num_qubits):
        vqc.h(q)


def _variational_layer(vqc, parameters_variational, num_qubits, depth):
    """ Variational layer
    """
    for q in range(num_qubits):
        idx = (2 * num_qubits) * depth + 2 * q
        vqc.rz(parameters_variational[idx], q)
        vqc.ry(parameters_variational[idx + 1], q)


def _encoding_layer_with_scaling(vqc, parameters_encoding, parameters_scaling, num_qubits, depth):
    """ Feature map with multiplicative scaling parameters
    """
    for q in range(num_qubits):
        idx = (2 * num_qubits) * depth + 2 * q
        vqc.ry(parameters_scaling[idx] * parameters_encoding[q], q)
        vqc.rz(parameters_scaling[idx+1] * parameters_encoding[q], q)


def _encoding_layer_with_idu(vqc, parameters_encoding, num_qubits, depth):
    """ Feature map incremental data uploading (as proposed in https://ieeexplore.ieee.org/document/9951318)
    """
    for q in range(num_qubits):
        idx = num_qubits * depth + q
        vqc.rx(parameters_encoding[idx], q)


def _encoding_layer(vqc, parameters_encoding, num_qubits):
    """ Feature map
    """
    for q in range(num_qubits):
        vqc.ry(parameters_encoding[q], q)
        vqc.rz(parameters_encoding[q], q)


def _entanglement_layer_nn_cx(vqc, num_qubits):
    """ CX Entangling layer (nearest neighbor)
    """
    for q in range(num_qubits-1):
        vqc.cx(q, q+1)
    if num_qubits > 2:
        vqc.cx(num_qubits-1, 0)


def _entanglement_layer_full_cz(vqc, num_qubits):
    """ CZ Entangling layer (all-to-all)
    """
    for q in range(num_qubits):
        for qq in range(q+1, num_qubits):
            vqc.cz(q, qq)


def generate_circuit(
        num_qubits: int = 4,
        depth: int = 1,
        entanglement_structure: str = 'nn',
        input_scaling: bool = False,
        incremental_data_uploading: bool = False
) -> tuple[QuantumCircuit, ParameterVector, tuple[ParameterVector, None | ParameterVector]]:
    """
    Generate variational circuit.

    Args:
        num_qubits: Number of qubits the VQC should act on
        depth: Number of layers of the VQC.
        entanglement_structure: Entanglement type, i.e. CNOT nearest-neighbors or CZ all-to-all.
        input_scaling: Use additional trainable input scaling parameters.
        incremental_data_uploading: Use incremental data uploading.
    """

    if input_scaling and incremental_data_uploading:
        raise ValueError('Input scaling and incremental data uploading can only be used separately.')

    if entanglement_structure not in ['nn', 'full']:
        ValueError('Entanglement structure {} unknown. Choose either `nn` (nearest-neighbors) or `full` (all-to-all)')

    vqc = QuantumCircuit(num_qubits)

    if incremental_data_uploading:
        # data is uploaded incrementally after each variational layer
        num_parameters_encoding = depth * num_qubits
    else:
        # data encoding (`depth` instances, i.e. data re-uploading for `depth>=2`)
        num_parameters_encoding = num_qubits
    parameters_encoding = ParameterVector('a', length=num_parameters_encoding)

    # variational parameters (`depth+1` instances)
    num_parameters_variational = 2 * num_qubits * (depth + 1)
    parameters_variational = ParameterVector('b', length=num_parameters_variational)

    # optional input scaling
    parameters_scaling = None
    if input_scaling:
        num_parameters_scaling = 2 * num_qubits * depth
        parameters_scaling = ParameterVector('c', length=num_parameters_scaling)

    # compose circuit
    _equal_superposition(vqc, num_qubits)
    vqc.barrier()
    for d in range(depth):
        _variational_layer(vqc, parameters_variational, num_qubits, depth=d)
        _entanglement_layer_nn_cx(vqc, num_qubits) if 'nn' == entanglement_structure else _entanglement_layer_full_cz(vqc, num_qubits)
        vqc.barrier()
        if parameters_scaling:
            _encoding_layer_with_scaling(vqc, parameters_encoding, parameters_scaling, num_qubits, depth=d)
        elif incremental_data_uploading:
            _encoding_layer_with_idu(vqc, parameters_encoding, num_qubits, depth=d)
        else:
            _encoding_layer(vqc, parameters_encoding, num_qubits)
        vqc.barrier()
    _variational_layer(vqc, parameters_variational, num_qubits, depth=depth)

    return vqc, parameters_encoding, (parameters_variational, parameters_scaling)
