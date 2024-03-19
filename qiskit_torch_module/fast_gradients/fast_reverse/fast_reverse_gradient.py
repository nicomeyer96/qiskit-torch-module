# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

###################################################################################################################
# This code is a modified version of qiskit_algorithms.gradients.reverse_gradient and aprt of Qiskit-Torch-Module #
# The main modifications include:                                                                                 #
# - An alternated input structure for _run()                                                                      #
# - Efficient gradient computation for multiple (not necessarily commuting) observables                           #
# - Parallelization over batch of input data                                                                      #
###################################################################################################################

"""Estimator gradients with the classically efficient reverse mode."""

import numpy as np
from collections.abc import Sequence
import logging
import multiprocessing as mp
from threadpoolctl import threadpool_limits

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator

from qiskit_algorithms.gradients.reverse.bind import bind
from qiskit_algorithms.gradients.reverse.derive_circuit import derive_circuit
from qiskit_algorithms.gradients.reverse.split_circuits import split
from qiskit_algorithms.gradients.utils import DerivativeType

from ..fast_base import FastBaseEstimatorGradient, FastEstimatorGradientResult


logger = logging.getLogger(__name__)


class FastReverseGradientEstimator(FastBaseEstimatorGradient):
    """Estimator gradients with the classically efficient reverse mode.

    .. note::

        This gradient implementation is based on statevector manipulations and scales
        exponentially with the number of qubits. However, for small system sizes it can be very fast
        compared to circuit-based gradients.

    This class implements the calculation of the expectation gradient as described in
    [1]. By keeping track of two statevectors and iteratively sweeping through each parameterized
    gate, this method scales only linearly with the number of parameters.

    **References:**

        [1]: Jones, T. and Gacon, J. "Efficient calculation of gradients in classical simulations
             of variational quantum algorithms" (2020).
             `arXiv:2009.02823 <https://arxiv.org/abs/2009.02823>`_.

    """

    def __init__(self, circuit: QuantumCircuit, derivative_type: DerivativeType = DerivativeType.REAL):
        """
        Args:
            circuit: The circuit to compute the gradients of in the _run() method
            derivative_type: Defines whether the real, imaginary or real plus imaginary part
                of the gradient is returned.
        """
        SUPPORTED_GATES = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]
        dummy_estimator = Estimator()  # this is required by the base class, but not used
        super().__init__(dummy_estimator, circuit, supported_gates=SUPPORTED_GATES, derivative_type=derivative_type)

    @FastBaseEstimatorGradient.derivative_type.setter
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def _run(
        self,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
        num_threads: int = 0,
        **options,
    ) -> FastEstimatorGradientResult:
        """Compute the gradients of the expectation values by the adjoint method."""

        # the metadata only contains the parameters as there are no run configs here
        metadata = (
            {
                "parameters": parameters,
                "derivative_type": self.derivative_type,
            }
        )

        # check for the use of multiprocessing
        if not 0 <= num_threads <= mp.cpu_count():
            raise ValueError(
                f"{num_threads} exceeds the number of available cores ({mp.cpu_count()})."
                f"Set ``num_threads`` to `0` to automatically use all available cores."
            )
        if 0 == num_threads:
            num_threads = mp.cpu_count()
        if not 1 == num_threads and len(parameter_values) < num_threads:
            num_threads = len(parameter_values)

        if 1 == num_threads:
            result = self._run_sequential(
                observables=observables,
                parameter_values=parameter_values,
                parameters=parameters,
                **options,
            )
            return FastEstimatorGradientResult(result, metadata)
        else:
            # determine which part of the batch to send to which worker
            batch_size = len(parameter_values)
            per_thread = batch_size // num_threads
            number_remaining = batch_size - per_thread * num_threads
            indices_mp = []
            parameter_values_mp = []
            start_index = 0
            for p in range(num_threads):
                end_index = start_index + per_thread
                if p < number_remaining:
                    end_index += 1
                indices_thread = list(range(start_index, end_index))
                indices_mp.append(indices_thread)
                parameter_values_thread = [parameter_values[ind] for ind in indices_thread]
                parameter_values_mp.append(parameter_values_thread)
                start_index = end_index

            # shared data structure for storing results
            result_mp = mp.Array('f', batch_size * len(observables) * len(parameters))

            # distribute tasks among workers
            process = [
                mp.Process(
                    target=self._run_parallel,
                    args=(
                        observables, pv_thread, parameters, result_mp, ind_thread
                    )
                )
                for pv_thread, ind_thread in zip(parameter_values_mp, indices_mp)
            ]
            for p in process:
                p.start()
            for p in process:
                p.join()

            # reconstruct from flattened shape
            result = np.array(result_mp, dtype=np.float32).reshape(-1, len(observables), len(parameters))
            del result_mp
            return FastEstimatorGradientResult(result, metadata)

    def _run_sequential(
            self,
            observables: Sequence[BaseOperator],
            parameter_values: Sequence[Sequence[float]],
            parameters: Sequence[Parameter],
            **options,
            ) -> Sequence[Sequence[Sequence[float]]]:

        g_parameter_values, g_parameters = self._preprocess(
            parameter_values, parameters
        )
        if self._deactivate_numpy_multithreading:
            with threadpool_limits(limits=1, user_api='blas'):
                results = self._run_unique(
                    observables, g_parameter_values, g_parameters, **options
                )
        else:
            results = self._run_unique(
                observables, g_parameter_values, g_parameters, **options
            )
        result_final = self._postprocess(results, parameter_values, parameters)
        return np.array(result_final, dtype=np.float32)

    def _run_parallel(
            self,
            observables: Sequence[BaseOperator],
            parameter_values: Sequence[Sequence[float]],
            parameters: Sequence[Parameter],
            mp_result: mp.Array = None,
            mp_index: Sequence[int] = None,
            **options,
            ) -> None:

        g_parameter_values, g_parameters = self._preprocess(
            parameter_values, parameters
        )
        if self._deactivate_numpy_multithreading:
            with threadpool_limits(limits=1, user_api='blas'):
                results = self._run_unique(
                    observables, g_parameter_values, g_parameters, **options
                )
        else:
            results = self._run_unique(
                observables, g_parameter_values, g_parameters, **options
            )
        result_final = self._postprocess(results, parameter_values, parameters)

        # store (flattened) results into shared data structure
        result_final_gradients = result_final
        size_per_input = len(observables) * len(parameters)
        for i, (ind, r) in enumerate(zip(mp_index, result_final_gradients)):
            r_flat = np.array(r).flatten()
            assert r_flat.shape[0] == size_per_input
            mp_result[ind * size_per_input:(ind + 1) * size_per_input] = r_flat

    def _run_unique(
        self,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Parameter],
        **options,  # pylint: disable=unused-argument
    ) -> Sequence[Sequence[Sequence[float]]]:

        num_gradients = len(parameter_values)
        gradients = []

        for i in range(num_gradients):
            # temporary variables for easier access
            circuit = self._gradient_circuit.gradient_circuit
            values = parameter_values[i]

            # keep track of the parameter order of the circuit, as the circuit splitting might
            # produce a list of unitaries in a different order
            # original_parameter_order = [p for p in circuit.parameters if p in parameters_]

            # split the circuit and generate lists of unitaries [U_1, U_2, ...] and
            # parameters [p_1, p_2, ...] in these unitaries
            unitaries, paramlist = split(circuit, parameters=parameters)

            parameter_binds = dict(zip(circuit.parameters, values))
            bound_circuit = bind(circuit, parameter_binds)

            # initialize state variables -- we use the same naming as in the paper
            phi = Statevector(bound_circuit)
            lams = [_evolve_by_operator(observable, phi) for observable in observables]

            # store gradients in a dictionary to return them in the correct order
            # important to do full dictionary generation for each observable instance to prevent call-by-reference
            gradss = [{param: 0j for param in parameters} for _ in observables]

            num_parameters = len(unitaries)
            for j in reversed(range(num_parameters)):
                unitary_j = unitaries[j]

                # We currently only support gates with a single parameter -- which is reflected
                # in self.SUPPORTED_GATES -- but generally we could also support gates with multiple
                # parameters per gate
                parameter_j = paramlist[j][0]

                # get the analytic gradient d U_j / d p_j and bind the gate
                deriv = derive_circuit(unitary_j, parameter_j)
                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

                # iterate the state variable
                unitary_j_dagger = bind(unitary_j, parameter_binds).inverse()
                phi = phi.evolve(unitary_j_dagger)

                # pre-compute here, as it is the same for all observables
                phis_evolved = [phi.evolve(gate).data for _, gate in deriv]
                grad = [
                    sum(
                        coeff * lam.conjugate().data.dot(phi_evolved) for (coeff, _), phi_evolved in zip(deriv, phis_evolved)
                    )
                    for lam in lams
                ]

                # Compute the full gradient (real and complex parts) as all information is available.
                # Later, based on the derivative type, cast to real/imag/complex.
                for o, g in enumerate(grad):
                    gradss[o][parameter_j] += g

                if j > 0:
                    lams = [lam.evolve(unitary_j_dagger) for lam in lams]

            gradient = [np.array(list(grads.values())) for grads in gradss]
            gradients.append(self._to_derivtype(gradient))

        return gradients

    def _to_derivtype(self, gradient):
        # this disable is needed as Pylint does not understand derivative_type is a property if
        # it is only defined in the base class and the getter is in the child
        # pylint: disable=comparison-with-callable
        if self.derivative_type == DerivativeType.REAL:
            return 2 * np.real(gradient)
        if self.derivative_type == DerivativeType.IMAG:
            return 2 * np.imag(gradient)

        return 2 * gradient


def _evolve_by_operator(operator, state):
    """Evolve the Statevector state by operator."""
    # try casting to sparse matrix and use sparse matrix-vector multiplication, which is
    # a lot faster than using Statevector.evolve
    try:
        spmatrix = operator.to_matrix(sparse=True)
        evolved = spmatrix @ state.data
        return Statevector(evolved)
    except (TypeError, AttributeError):
        logger.info("Operator is not castable to a sparse matrix, using Statevector.evolve.")
    return state.evolve(operator)
