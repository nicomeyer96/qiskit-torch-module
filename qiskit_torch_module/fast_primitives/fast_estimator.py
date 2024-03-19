# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

##################################################################################################
# This code is a modified version of qiskit.primitives.estimator and part of Qiskit-Torch-Module #
# The main modifications include:                                                                #
# - An alternated input structure for _run()                                                     #
# - Efficient gradient computation for multiple (not necessarily commuting) observables          #
# - Parallelization over batch of input data                                                     #
##################################################################################################

"""
FastEstimator class
"""

import numpy as np
from collections.abc import Sequence
from typing import Any
from importlib.metadata import version
import multiprocessing as mp
import warnings
from threadpoolctl import threadpool_limits

from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import bound_circuit_to_instruction

from .fast_base import FastBaseEstimator, FastEstimatorResult


class FastEstimator(FastBaseEstimator[PrimitiveJob[FastEstimatorResult]]):
    """
    Reference implementation of :class:`FastBaseEstimator`.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
    """

    def __init__(self, circuit, *, options: dict | None = None):
        """
        Args:
            circuit: The circuit to evaluate the expectation values of
            options: Default options.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(circuit, options=options)

    def _call_sequential(
            self,
            observables: Sequence[BaseOperator],
            parameter_values: Sequence[Sequence[float]],
            shots: int | None = None,
            seed: int | None = None,
            **run_options,
    ) -> FastEstimatorResult:
        if self._deactivate_numpy_multithreading:
            with threadpool_limits(limits=1, user_api='blas'):
                return self._call(observables, parameter_values, shots, seed, **run_options)
        else:
            return self._call(observables, parameter_values, shots, seed, **run_options)

    def _call_parallel(
            self,
            observables: Sequence[BaseOperator],
            parameter_values: Sequence[Sequence[float]],
            num_threads: int = 0,
            shots: int | None = None,
            seed: int | None = None,
            **run_options,
    ) -> FastEstimatorResult:

        if self._circuit.num_qubits > 18:
            warnings.warn('Depending on your system batch-parallelization might struggle with allocation of resources '
                          'for large systems above about 18 qubits. '
                          'Consider using sequential processing (``num_threads=1``)')

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
        result_mp = mp.Array('f', batch_size * len(observables))
        shots_mp, variance_mp = None, None
        if shots is not None:
            shots_mp = mp.Array('f', batch_size * len(observables))
            variance_mp = mp.Array('f', batch_size * len(observables))

        # distribute tasks among workers
        process = [
            mp.Process(
                target=self._call_parallel_,
                args=(
                    observables, pv_thread, shots, seed, result_mp, shots_mp, variance_mp, ind_thread
                )
            )
            for pv_thread, ind_thread in zip(parameter_values_mp, indices_mp)
        ]
        for p in process:
            p.start()
        for p in process:
            p.join()

        # reconstruct from flattened shape
        result = np.array(result_mp, dtype=np.float32).reshape(-1, len(observables))
        if shots is not None:
            shots = np.array(shots_mp).astype(int).reshape(-1, len(observables))
            variance = np.array(variance_mp, dtype=np.float32).reshape(-1, len(observables))
            metadata = [[{'shots': s_, 'variance': v_} for s_, v_ in zip(s, v)] for s, v in zip(shots, variance)]
        else:
            metadata = [[{} for _ in range(len(observables))] for _ in range(len(parameter_values))]

        del result_mp
        return FastEstimatorResult(result, metadata)

    def _call_parallel_(
            self,
            observables: Sequence[BaseOperator],
            parameter_values: Sequence[Sequence[float]],
            shots: int | None = None,
            seed: int | None = None,
            mp_result: mp.Array = None,
            mp_shots: mp.Array = None,
            mp_variance: mp.Array = None,
            mp_index: Sequence[int] = None
    ) -> None:

        if self._deactivate_numpy_multithreading:
            with threadpool_limits(limits=1, user_api='blas'):
                result = self._call(observables, parameter_values, shots, seed)
        else:
            result = self._call(observables, parameter_values, shots, seed)

        # store (flattened) results into shared data structure
        result_values = result.values
        size_per_input = len(observables)
        for i, (ind, r) in enumerate(zip(mp_index, result_values)):
            assert r.shape[0] == size_per_input
            mp_result[ind * size_per_input:(ind + 1) * size_per_input] = r
        if shots is not None:
            result_metadata = result.metadata
            for i, (ind, m) in enumerate(zip(mp_index, result_metadata)):
                for j, m_ in enumerate(m):
                    mp_variance[ind * size_per_input + j] = m_['variance']
                    mp_shots[ind * size_per_input + j] = m_['shots']

    def _call(
        self,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
        **run_options,
    ) -> FastEstimatorResult:
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[list[dict[str, Any]]] = [[{} for _ in range(len(observables))] for _ in range(len(parameter_values))]
        expectation_values = []
        for parameter_values_, metadatum in zip(parameter_values, metadata):
            # bind parameters to circuit
            # CAUTION: Qiskit always does this in an alphabetical order!
            bound_circuit = self._circuit \
                if 0 == len(self._circuit.parameters) \
                else self._circuit.assign_parameters(dict(zip(self._circuit.parameters, parameter_values_)))

            final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
            expectation_values_ = [final_state.expectation_value(obs) for obs in observables]

            if shots is None:
                expectation_values.append(expectation_values_)
            else:
                expectation_values__ = []
                for obs_, expectation_value_, metadatum_ in zip(observables, expectation_values_, metadatum):
                    expectation_value = np.real_if_close(expectation_value_)
                    sq_obs = (obs_ @ obs_).simplify(atol=0)
                    sq_exp_val = np.real_if_close(final_state.expectation_value(sq_obs))
                    variance = sq_exp_val - expectation_value ** 2
                    variance = (max(variance, 0)).astype(np.float32)
                    standard_error = np.sqrt(variance / shots)
                    expectation_value_with_error = rng.normal(expectation_value, standard_error)
                    expectation_values__.append(expectation_value_with_error)
                    metadatum_["shots"] = shots
                    metadatum_["variance"] = variance
                expectation_values.append(expectation_values__)

        return FastEstimatorResult(np.real_if_close(np.array(expectation_values)), metadata)

    def _run(
        self,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        num_threads: int = 0,
        **run_options,
    ):

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
        #
        # with threadpool_limits(limits=1, user_api='blas'):
        if 1 == num_threads:
            # run sequential
            job = PrimitiveJob(
                self._call_sequential, observables, parameter_values,
                run_options.pop("shots", None), run_options.pop("seed", None)
            )
            # account for different syntax in qiskit versions 0.x and 1.x
            if '0' == version('qiskit')[0]:
                job.submit()
            else:
                job._submit()
            return job
        else:
            # run (batch-)parallel
            job = PrimitiveJob(
                self._call_parallel, observables, parameter_values, num_threads,
                run_options.pop("shots", None), run_options.pop("seed", None)
            )
            # account for different syntax in qiskit versions 0.x and 1.x
            if '0' == version('qiskit')[0]:
                job.submit()
            else:
                job._submit()
            return job
