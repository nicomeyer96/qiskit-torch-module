# This code is part of Qiskit-Torch-Module.
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

import warnings
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression


def generate_alphabetically_ordered_circuit(
        circuit: QuantumCircuit,
        encoding_params: Sequence[Parameter],
        variational_params: Sequence[Sequence[Parameter]]
) -> tuple[QuantumCircuit, Sequence[Parameter], Sequence[Sequence[Parameter]]]:
    """
    Method for generating circuit with consistent alphabetical ordering (as qiskit binds parameters in this way),
    w.r.t. the order defined by the sets of encoding and variational parameters.

        Args:
            circuit: Raw quantum circuit ansatz
            encoding_params: Raw parameters used for data encoding, must be present in circuit
            variational_params: Raw parameters used for training, must be present in circuit

        Returns:
            Alphabetically ordered version of circuit and parameters

        Raises:
            ValueError: Invalid combination of circuit and parameters
    """
    # prevent inplace modification of original circuit
    circuit = circuit.copy()
    # check if all provided parameters also are present in the circuit (reverse is automatically checked below)
    params_circuit = circuit.parameters
    for p_ in encoding_params:
        if p_ not in params_circuit:
            raise ValueError('The parameter `{}` does not appear in the provided circuit.'.format(p_))
    for vp_ in variational_params:
        for p_ in vp_:
            if p_ not in params_circuit:
                raise ValueError('The parameter `{}` does not appear in the provided circuit.'.format(p_))

    # construct new Parameter sets with `fixed` naming
    digits_encoding = len(str(len(encoding_params) - 1))
    encoding_params_ = [Parameter('a_{}'.format(str(i).zfill(digits_encoding))) for i in range(len(encoding_params))]
    variational_params_ = []
    if len(variational_params_) > 25:
        warnings.warn("Using more than 25 distinct parameter sets might lead to unexpected behaviour, as the naming is "
                      "chosen as ASCII characters from `b` to `z`")
    for set_index, vp in enumerate(variational_params):
        digits_variational = len(str(len(vp) - 1))
        variational_params_.append([Parameter('{}_{}'.format(chr(ord('b') + set_index),
                                                             str(index).zfill(digits_variational))) for index in
                                    range(len(vp))])

    def _find_parameter(_param: Parameter) -> dict[Parameter: Parameter]:
        """ Locate the parameter object in original input, construct dictionary for renaming

            Args:
                _param: Parameter to search for

            Returns:
                Mapping of found parameter to new naming

            Raises:
                Value Error: Parameter not present in provided parameter set
        """
        # method for finding the mapping of original parameter to parameter with modified naming
        _indices_enc = [_index for _index, _p in enumerate(encoding_params) if _p == _param]
        _indices_var = [[_index for _index, _p in enumerate(_vp) if _p == _param] for _vp in variational_params]
        if len(_indices_enc) + sum([len(_iv) for _iv in _indices_var]) > 1:
            raise ValueError('The circuit parameter `{}` appears multiple times in the provided '
                             'parameter sets'.format(_param))
        if 0 == len(_indices_enc) + sum([len(_iv) for _iv in _indices_var]):
            raise ValueError('The circuit parameter `{}` does not appear in the provided '
                             'parameter sets'.format(_param))
        if len(_indices_enc) > 0:
            # parameter is used for encoding
            return {encoding_params[_indices_enc[0]]: encoding_params_[_indices_enc[0]]}
        else:
            # determine to which variational parameter set the current parameter belongs
            _index_set = [i for i, _iv in enumerate(_indices_var) if len(_iv) > 0][0]
            return {variational_params[_index_set][_indices_var[_index_set][0]]:
                        variational_params_[_index_set][_indices_var[_index_set][0]]}

    circuit_ = circuit.copy_empty_like()
    # de-construct circuit
    for instruction in circuit.data:
        if instruction.operation.is_parameterized():
            params_ = []
            for param in instruction.operation.params:
                subs = {}
                if isinstance(param, Parameter):
                    # single Parameter element
                    subs.update(_find_parameter(param))
                else:
                    # ParameterExpressions (with potentially multiple Parameter elements)
                    for param_single in param.parameters:
                        subs.update(_find_parameter(param_single))
                # substitute old parameters for new ones
                param_ = param.subs(subs)
                params_.append(param_)
            instruction.operation.params = params_
        # re-construct circuit
        circuit_.append(instruction.operation, instruction.qubits, instruction.clbits)
    if isinstance(circuit.global_phase, ParameterExpression):
        raise RuntimeError('This method currently does not support ParameterExpressions for the global phase.')
    return circuit_, encoding_params_, variational_params_
