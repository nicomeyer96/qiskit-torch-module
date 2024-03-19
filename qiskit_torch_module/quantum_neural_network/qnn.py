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

import numpy as np
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .utils import generate_alphabetically_ordered_circuit
from ..fast_primitives import FastEstimator
from ..fast_gradients import FastReverseGradientEstimator


class QNN:
    """ This class implements a quantum neural network, which combines forward and backward class for a given setup.

        Args:
            circuit: Quantum circuit ansatz
            encoding_params: Parameters used for data encoding, must be present in circuit | None if the circuit does
                not use data encoding
            variational_params: Parameters used for training, must be present in circuit
            observables: Observables to evaluate, corresponds to output of QNN (default: Pauli-Z on all qubits)
            num_threads_forward: Number of parallel threads for forward computation (default: all available threads)
            num_threads_backward: Number of parallel threads for backward computation (default: all available threads)
    """

    def __init__(
            self,
            circuit: QuantumCircuit,
            encoding_params: Sequence[Parameter] | ParameterVector | None,
            variational_params: Sequence[Sequence[Parameter]] | Sequence[ParameterVector] | Sequence[Parameter] | ParameterVector,
            observables: Sequence[BaseOperator] | BaseOperator | str = 'individualZ',
            num_threads_forward: int = 0,
            num_threads_backward: int = 0,
    ):

        # singleton variational parameter set
        if (isinstance(variational_params[0], (Parameter, ParameterExpression))
                or isinstance(variational_params, ParameterVector)):
            variational_params = (variational_params, )

        self._circuit, self._encoding_params, self._variational_params =\
            generate_alphabetically_ordered_circuit(circuit, encoding_params, variational_params)

        # Estimators for estimation of expectation values and gradients
        self._estimator_expval = FastEstimator(self._circuit)
        self._estimator_gradient = FastReverseGradientEstimator(self._circuit)

        self._num_qubits = self._circuit.num_qubits

        self._observables = self._validate_and_preprocess_observables(observables)

        self._num_threads_forward = num_threads_forward
        self._num_threads_backward = num_threads_backward

        self._num_encoding_params = len(self._encoding_params)
        self._num_variational_params = [len(_vp) for _vp in self._variational_params]
        self._num_variational_param_sets = len(self._num_variational_params)

    def forward(
            self,
            encoding_data: Sequence[Sequence[float]] | Sequence[float] | None,
            variational_weights: Sequence[Sequence[float]] | Sequence[float],
    ) -> Sequence[Sequence[float]]:
        """ Realizes forward pass of QNN, i.e. computation of expectation values

            Args:
                encoding_data: Input to the QNN
                variational_weights: Values for trainable weights of the QNN

            Returns:
                Result of forward pass (i.e. expectation values)
        """
        combined_data = self._validate_and_preprocess_data(encoding_data, variational_weights)
        job = self._estimator_expval.run(
            observables=self._observables,
            parameter_values=combined_data,
            num_threads=self._num_threads_forward
        )
        return job.result().values

    def backward(
            self,
            encoding_data: Sequence[Sequence[float]] | Sequence[float] | None,
            variational_weights: Sequence[Sequence[float]] | Sequence[float],
            encoding_gradients: bool = False,
            variational_gradients: Sequence[bool] | bool = True
    ) -> tuple[Sequence[Sequence[Sequence[float]]] | None,
               Sequence[Sequence[Sequence[Sequence[float]]] | None] | Sequence[Sequence[Sequence[float]]] | None]:
        """ Realizes backward pass of QNN, i.e. computation of gradients w.r.t. variational parameters

            Args:
                encoding_data: Input to the QNN
                variational_weights: Values for trainable weights of the QNN
                encoding_gradients: Whether to compute gradients w.r.t. input parameters (default: False)
                variational_gradients: Whether to compute gradients w.r.t. trainable parameter sets (default: all True)

            Returns:
                Result of backward pass (i.e. gradients w.r.t. variational parameters)
        """
        combined_data = self._validate_and_preprocess_data(encoding_data, variational_weights)
        parameters = self._validate_and_preprocess_parameters(encoding_gradients, variational_gradients)
        job = self._estimator_gradient.run(
            observables=self._observables,
            parameter_values=combined_data,
            parameters=parameters,
            num_threads=self._num_threads_backward
        )
        gradients = job.result().gradients
        # cut into gradients w.r.t. input and variational parameters
        return self._postprocess_gradients(gradients, encoding_gradients, variational_gradients)

    def circuit(self) -> QuantumCircuit:
        """ Return (pre-processed, i.e. alphabetically ordered) quantum circuit
        """
        return self._circuit

    def parameters(self) -> tuple[Sequence[Parameter], Sequence[Sequence[Parameter]]]:
        """ Return (pre-processed, i.e. alphabetically ordered) encoding and variational parameters
        """
        return self._encoding_params, self._variational_params

    def input_parameters(self) -> Sequence[Parameter]:
        """ Return (pre-processed, i.e. alphabetically ordered) encoding parameters
        """
        return self._encoding_params

    def trainable_parameters(self) -> Sequence[Sequence[Parameter]]:
        """ Return (pre-processed, i.e. alphabetically ordered) variational parameters
        """
        return self._variational_params

    def num_parameters(self) -> tuple[int, Sequence[int]]:
        """ Return number of encoding and variational parameters (per parameter set)
        """
        return self._num_encoding_params, self._num_variational_params

    def num_input_parameters(self) -> int:
        """ Return number of encoding parameters
        """
        return self._num_encoding_params

    def num_trainable_parameters(self) -> Sequence[int]:
        """ Return number of variational parameters (per parameter set)
        """
        return self._num_variational_params

    def observables(self) -> Sequence[BaseOperator]:
        """ Return observables
        """
        return self._observables

    def num_observables(self) -> int:
        """ Return number of observables
        """
        return len(self._observables)

    def num_threads_forward(self) -> int:
        """ Return number of threads used for forward pass (`0` means all available)
        """
        return self._num_threads_forward

    def num_threads_backward(self) -> int:
        """ Return number of threads used for backward pass (`0` means all available)
        """
        return self._num_threads_backward

    def set_num_threads_forward(self, num_threads_forward: int):
        """ Set number of threads used for forward pass (`0` means all available)
        """
        self._num_threads_forward = num_threads_forward

    def set_num_threads_backward(self, num_threads_backward: int):
        """ Set number of threads used for backward pass (`0` means all available)
        """
        self._num_threads_backward = num_threads_backward

    def _validate_and_preprocess_data(
            self,
            encoding_data: Sequence[Sequence[float]] | Sequence[float] | None,
            variational_weights: Sequence[Sequence[float]] | Sequence[float]
    ) -> Sequence[Sequence[float]]:
        """ Validate shape etc. of input, preprocess for forward and backward pass

        Args:
            encoding_data: Input to the QNN
            variational_weights: Values for trainable weights of the QNN

        Returns:
            Concatenated input data to be bound to encoding and variational parameters

        Raises:
            ValueError: Mismatch of size / dimensions for provided input and underlying circuit
        """
        # handle case of no input data
        if encoding_data is None:
            _encoding_data = np.array([[None]])
        # handle singleton encoding data set
        elif isinstance(encoding_data[0], (float, np.floating)):
            _encoding_data = np.array((encoding_data, ))
        else:
            _encoding_data = np.array(encoding_data)
        # validate encoding data
        if encoding_data is not None and not self._num_encoding_params == _encoding_data.shape[1]:
            raise ValueError(
                "The size of the provided encoding data ({}) does not correspond to the number of encoding parameters "
                "in the circuit ({}).".format(len(encoding_data), self._num_encoding_params))
        # validate variational parameters and potentially do some re-shaping
        if isinstance(variational_weights[0], (float, np.floating)):
            # all variational parameters are given in one Sequence
            if not sum(self._num_variational_params) == len(variational_weights):
                raise ValueError(
                    "The size of the provided variational data ({}) does not correspond to the number of variational "
                    "parameters in the circuit ({}).".format(len(variational_weights), sum(self._num_variational_params)))
            _variational_weights = np.array(variational_weights)
        else:
            # the variational parameters are given in per-parameter-set form
            for set_index, (variational_weights_, num_variational_params_) in enumerate(zip(variational_weights, self._num_variational_params)):
                if not num_variational_params_ == len(variational_weights_):
                    raise ValueError(
                        "The size of the provided variational data ({}) for parameter set #{} does not correspond "
                        "to the number of variational parameters for this set in the circuit ({})."
                        .format(len(variational_weights), set_index, sum(self._num_variational_params)))

            # reshape in case variational data is given in per-parameter-set form
            _variational_weights = np.concatenate(variational_weights).ravel()
        # stack variational parameters a sufficient number of times and concatenate
        return np.concatenate((_encoding_data, np.tile(_variational_weights, (len(_encoding_data), 1))), axis=1)

    def _validate_and_preprocess_observables(
            self,
            observables: Sequence[BaseOperator] | BaseOperator | str
    ) -> Sequence[BaseOperator]:
        """ Validate shape etc. of observables, preprocess for forward and backward pass

        Args:
            observables: Observables to evaluate on the circuit

        Returns:
            Pre-processed list of observables

        Raises:
            Value Error: Invalid observable was provided
        """
        if isinstance(observables, str):
            if 'individualZ' == observables:
                return [SparsePauliOp(i*'I' + 'Z' + (self._num_qubits-i-1)*'I') for i in range(self._num_qubits)]
            elif 'tensoredZ' == observables:
                return (SparsePauliOp(self._num_qubits * 'Z'), )
            else:
                raise ValueError('The observable `{}` is not implemented. '
                                 'Choose either `individualZ` for single-qubit Pauli-Z observables on all qubits,'
                                 '`tensoredZ` for a tensored Pauli-Z observable on all qubits,'
                                 'or explicitly provide an observable / a list of observables.'.format(observables))
        elif isinstance(observables, BaseOperator):
            # singleton observable
            return (observables, )
        else:
            return observables

    def _validate_and_preprocess_parameters(
            self,
            encoding_gradients: bool,
            variational_gradients: Sequence[bool] | bool
    ) -> Sequence[Parameter]:
        """ Validate shape etc. of gradient computation flags, preprocess for forward and backward pass

        Args:
            encoding_gradients: Whether to compute gradients w.r.t. input parameters
            variational_gradients: Whether to compute gradients w.r.t. trainable parameter sets

        Returns:
            List of parameters of which to compute the gradients of

        Raises:
            ValueError: Trying to acquire gradients w.r.t. zero variational parameters
        """
        if isinstance(variational_gradients, bool):
            # singleton variational_gradients flag
            if not (encoding_gradients or variational_gradients):
                raise ValueError('The gradients of at least one set of parameters have to be computed.')
            variational_gradients = [True for _ in range(self._num_variational_param_sets)]
        else:
            if not (encoding_gradients or any(variational_gradients)):
                raise ValueError('The gradients of at least one set of parameters have to be computed.')
        # determine the set of parameters to take gradients of
        parameters = []
        if encoding_gradients:
            parameters.extend(self._encoding_params)
        for index_set, variational_gradients_ in enumerate(variational_gradients):
            if variational_gradients_:
                parameters.extend(self._variational_params[index_set])
        return parameters

    def _postprocess_gradients(
            self,
            gradients: Sequence[Sequence[Sequence[float]]],
            encoding_gradients: bool,
            variational_gradients: Sequence[bool] | bool
    ) -> tuple[Sequence[Sequence[Sequence[float]]] | None,
               Sequence[Sequence[Sequence[Sequence[float]]] | None] | Sequence[Sequence[Sequence[float]]] | None]:
        """ Postprocess gradients from backward pass, potentially reshape

            Args:
                gradients: Gradient values from backward pass
                encoding_gradients: Whether to compute gradients w.r.t. input parameters
                variational_gradients: Whether to compute gradients w.r.t. trainable parameter sets
                    Note: If given as a single value, the gradients w.r.t. all trainable parameters will be returned in
                    flattened version. If given as a list, the same gradients are given as list with set-associated elements.

            Returns:
                Gradients w.r.t. encoding parameters, gradients w.r.t. variational parameters either in flattened or per-parameter form.
        """
        cutting_index = 0
        if encoding_gradients:
            gradients_encoding = gradients[:, :, :self._num_encoding_params]
            cutting_index += self._num_encoding_params
        else:
            gradients_encoding = None
        if isinstance(variational_gradients, bool):
            # singleton variational_gradients flag -> return computed gradients in flattened form
            if variational_gradients:
                gradients_variational = gradients[:, :, cutting_index:]
            else:
                gradients_variational = None
        else:
            # variational_gradients flag were give parameter-set-wise -> return gradients also in this form
            gradients_variational = []
            for index_set, variational_gradients_ in enumerate(variational_gradients):
                if variational_gradients_:
                    gradients_variational.append(gradients[:, :, cutting_index:cutting_index+self._num_variational_params[index_set]])
                    cutting_index += self._num_variational_params[index_set]
                else:
                    gradients_variational.append(None)
        return gradients_encoding, gradients_variational
