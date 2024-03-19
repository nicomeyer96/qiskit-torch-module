# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

###############################################################################################################################
# This code is a modified version of qiskit_algorithms.gradients.base.base_estimator_gradient and part of Qiskit-Torch-Module #
# The main modifications include:                                                                                             #
# - An alternated input structure for run()                                                                                   #
# - Compatibility changed to align with ..fast_reverse.fast_reverse_gradient                                                  #
# - Possibility to deactivate automatic numpy multithreading (interferes with batch-parallelization)                          #
###############################################################################################################################

"""
Abstract base class of fast gradient class.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passes import TranslateParameterizedGates

from qiskit_algorithms.gradients.utils import (
    DerivativeType,
    _assign_unique_parameters,
    _make_gradient_parameters,
    _make_gradient_parameter_values
)
from qiskit_algorithms.algorithm_job import AlgorithmJob


class FastBaseEstimatorGradient(ABC):
    """Base class for an ``FastEstimatorGradient`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        circuit: QuantumCircuit,
        deactivate_numpy_multithreading: bool = None,
        supported_gates: Sequence[str] = None,
        options: Options | None = None,
        derivative_type: DerivativeType = DerivativeType.REAL,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            circuit: The circuit to evaluate the gradients of.
            deactivate_numpy_multithreading: Deactivate automated numpy multithreading
                (significantly decreases performance for >=14 qubits when using batch-parallelization)
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``.

                    - ``DerivativeType.REAL`` computes :math:`2 \mathrm{Re}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.IMAG`` computes :math:`2 \mathrm{Im}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.COMPLEX`` computes :math:`2 ⟨ψ(ω)|O(θ)|dω ψ(ω)〉`.

                Defaults to ``DerivativeType.REAL``, as this yields e.g. the commonly-used energy
                gradient and this type is the only supported type for function-level schemes like
                finite difference.
        """
        self._estimator: BaseEstimator = estimator
        if deactivate_numpy_multithreading is None:
            if circuit.num_qubits >= 14:
                # For 14 or more qubits the automatic parallelization of numpy methods like e.g. np.dot() in
                # Statevector-evaluation might interfere with batch-parallelization and decrease performance
                self._deactivate_numpy_multithreading = True
            else:
                self._deactivate_numpy_multithreading = False
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._derivative_type = derivative_type

        if not circuit.num_parameters:
            raise ValueError(f"The circuit is not parameterised.")
        self._circuit = circuit
        self._gradient_circuit = self._preprocess_circuit(supported_gates)

    @property
    def derivative_type(self) -> DerivativeType:
        """Return the derivative type (real, imaginary or complex).

        Returns:
            The derivative type.
        """
        return self._derivative_type

    def run(
        self,
        observables: Sequence[BaseOperator] | BaseOperator,
        parameter_values: Sequence[Sequence[float]] | Sequence[float],
        parameters: Sequence[Parameter] | None = None,
        **options,
    ) -> AlgorithmJob:
        """Run the job of the estimator gradient on the given circuits.

        Args:
            observables: The (list of) observable(s).
            parameter_values: The list (of list) of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of
                the specified parameters. Defaults to None, which means that the gradients of all parameters in
                the circuit are calculated.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Returns:
            The job object of the gradients of the expectation values. The [i, j, k]-th result corresponds to
            the gradient of the circuit w.r.t. the k-th parameter ``parameters[k]``, for the j-th observable
            ``observables[j]``, bound with the i-th value ``parameter_values[i, k]``.

        Raises:
            ValueError: Invalid arguments are given.
        """

        if isinstance(observables, BaseOperator):
            # Allow a single observable to be passed in.
            observables = (observables,)

        # handle setup with no input parameters
        if parameter_values[0] is None:
            parameter_values = parameter_values[1:]
        if parameter_values[0][0] is None:
            parameter_values = parameter_values[:, 1:]

        if isinstance(parameter_values[0], (float, np.floating)):
            # Allow for a single set of inputs
            parameter_values = (parameter_values,)

        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameters = self._circuit.parameters
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameters = parameters if parameters is not None else self._circuit.parameters

        # Validate the arguments.
        self._validate_arguments(self._circuit, observables, parameter_values, parameters)

        # The priority of run option is as follows:
        # options in ``run`` method > gradient's default options > primitive's default setting.
        opts = copy(self._default_options)
        opts.update_options(**options)

        # Run the job.
        job = AlgorithmJob(
            self._run, observables, parameter_values, parameters, **opts.__dict__
        )
        # Submit the job
        job.submit()
        return job

    @abstractmethod
    def _run(
        self,
        # circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
        **options,
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Compute the estimator gradients on the given circuits."""
        raise NotImplementedError()

    def _preprocess_circuit(self, supported_gates: Sequence[str]):
        translator = TranslateParameterizedGates(supported_gates)
        unrolled = translator(self._circuit)
        gradient_circuit = _assign_unique_parameters(unrolled)
        return gradient_circuit

    def _preprocess(
        self,
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
    ) -> tuple[Sequence[Sequence[float]], Sequence[Parameter]]:
        """Preprocess the gradient. This makes a gradient circuit for each circuit. The gradient
        circuit is a transpiled circuit by using the supported gates, and has unique parameters.
        ``parameter_values`` and ``parameters`` are also updated to match the gradient circuit.

        Args:
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Returns:
            The list of gradient circuits, the list of parameter values, and the list of parameters.
            parameter_values and parameters are updated to match the gradient circuit.
        """

        g_parameter_values = []
        for parameter_value_ in parameter_values:
            g_parameter_values.append(
                _make_gradient_parameter_values(self._circuit, self._gradient_circuit, parameter_value_)
            )
        g_parameters = _make_gradient_parameters(self._gradient_circuit, parameters)
        return g_parameter_values, g_parameters

    def _postprocess(
        self,
        results: Sequence[Sequence[Sequence[float]]],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Postprocess the gradients. This method computes the gradient of the original circuits
        by applying the chain rule to the gradient of the circuits with unique parameters.

        Args:
            results: The computed gradients for the circuits with unique parameters.
            parameter_values: The list of parameter values to be bound to the circuits.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Returns:
            The gradients of the original circuits.
        """
        g_parameters = _make_gradient_parameters(self._gradient_circuit, parameters)
        g_parameter_indices = {param: i for i, param in enumerate(g_parameters)}
        gradients = []
        num_observables = len(results[0])
        for idx, parameter_values_ in enumerate(parameter_values):
            gradient = np.zeros((num_observables, len(parameters)))
            if self.derivative_type == DerivativeType.COMPLEX:
                # If the derivative type is complex, cast the gradient to complex.
                gradient = gradient.astype("complex")
            for i, parameter in enumerate(parameters):
                for g_parameter, coeff in self._gradient_circuit.parameter_map[parameter]:
                    # Compute the coefficient
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values_[self._circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    # The original gradient is a sum of the gradients of the parameters in the
                    # gradient circuit multiplied by the coefficients.
                    for o in range(num_observables):
                        gradient[o][i] += (
                            float(bound_coeff)
                            * results[idx][o][g_parameter_indices[g_parameter]]
                        )
            gradients.append(gradient)
        return gradients

    @staticmethod
    def _validate_arguments(
        circuit: QuantumCircuit,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuit: The quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Raises:
            ValueError: Invalid arguments are given.
        """

        # handle setup with no input parameters
        if parameter_values[0] is None:
            parameter_values = parameter_values[1:]
        if parameter_values[0][0] is None:
            parameter_values = parameter_values[:, 1:]

        for i, parameter_values_ in enumerate(parameter_values):
            if len(parameter_values_) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_values_)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the circuit."
                )

        for i, observable_ in enumerate(observables):
            if circuit.num_qubits != observable_.num_qubits:
                raise ValueError(
                    f"The number of qubits of the circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable_.num_qubits})."
                )

        if not set(parameters).issubset(circuit.parameters):
            raise ValueError(
                "The parameters contains parameters not present in the circuit."
            )

    def activate_numpy_multithreading(self):
        """
        Activates the use of numpy-multithreading (e.g. for np.dot() in Statevector-evaluation). Will be used for
        system sizes of about >=14 qubits, might slow down batch-parallelization significantly. By default, it will
        be deactivated.
        """
        self._deactivate_numpy_multithreading = False

    def deactivate_numpy_multithreading(self):
        """
        Deactivates the use of numpy-multithreading (e.g. for np.dot() in Statevector-evaluation). Will be used for
        system sizes of about >=14 qubits, might slow down batch-parallelization significantly. By default, it will
        be deactivated.
        """
        self._deactivate_numpy_multithreading = True

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and gradient default options,
        where, if the same field is set in both, the gradient's default options override
        the primitive's default setting.

        Returns:
            The gradient default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the gradient's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the gradient default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The gradient default + estimator + run options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**options)
        return opts
