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

############################################################################################################
# This code is a modified version of qiskit.primitives.base.base_estimator and part of Qiskit-Torch-Module #
# The main modifications include:                                                                          #
# - An alternated input structure for run()                                                                #
# - Compatibility changed to align with ..fast_estimator                                                   #
# - Possibility to deactivate automatic numpy multithreading (interferes with batch-parallelization)       #
############################################################################################################

r"""

.. estimator-desc:

=====================
Overview of FastEstimator
=====================

Estimator class estimates expectation values of quantum circuits and observables.

An estimator is initialized with a quantum circuit (:math:`psi`). The estimator is used to
create a :class:`~qiskit.providers.JobV1`, via the
:meth:`fast_primitives.FastEstimator.run()` method. This method is called
with the following parameters

* observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.BaseOperator`
  objects. All observable(s) will be evaluated for all parameter sets in ``parameter_values``

* parameter values (:math:`\theta_k`): list of sets of values
  to be bound to the parameters of the quantum circuits
  (Sequence of Set of floats). All set of parameter values will be evaluated for all ``observables``.

The method returns a :class:`~qiskit.providers.JobV1` object, calling
:meth:`qiskit.providers.JobV1.result()` yields the list of expectation values
plus optional metadata like confidence intervals for the estimation.

.. math::

    \langle\psi(\theta_k)|H_j|\psi(\theta_k)\rangle

Here is an example of how the estimator is used.

.. code-block:: python

    from fast_primitives import FastEstimator
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    psi = RealAmplitudes(num_qubits=2, reps=2)

    H1 = SparsePauliOp.from_list([("IZ", 1)])
    H2 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

    theta1 = [0, 1, 2, 3, 4, 5]
    theta2 = [1, 2, 3, 4, 5, 6]
    theta3 = [2, 3, 4, 5, 6, 7]

    estimator = Estimator(psi)

    # calculate [ [ <psi(theta1)|H1|psi(theta1)> ] ]
    job = estimator.run(H1, theta1)
    job_result = job.result() # It will block until the job finishes.
    print(f"The primitive-job finished with result {job_result}"))

    # calculate [ [ <psi(theta1)|H1|psi(theta1)>,
    #               <psi(theta1)|H2|psi(theta1)> ],
    #             [ <psi(theta2)|H1|psi(theta2)>,
    #               <psi(theta2)|H2|psi(theta2)> ],
    #             [ <psi(theta3)|H1|psi(theta3)>,
    #               <psi(theta3)|H2|psi(theta3)> ] ]
    job2 = estimator.run([H1, H2], [theta1, theta2, theta3])
    job_result = job2.result()
    print(f"The primitive-job finished with result {job_result}")
"""

import numpy as np
from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar
from copy import copy

from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .fast_base_primitive import FastBasePrimitive

T = TypeVar("T", bound=Job)


class FastBaseEstimator(FastBasePrimitive, Generic[T]):
    """FastEstimator base class.

    Base class for FastEstimator that estimates expectation values of quantum circuits and observables.
    """

    __hash__ = None

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        deactivate_numpy_multithreading: bool = None,
        options: dict | None = None,
    ):
        """
        Creating an instance of an FastEstimator.

        Args:
            circuit: Quantum circuit to evaluate the expectation values of
            deactivate_numpy_multithreading: Deactivate automated numpy multithreading
                (significantly decreases performance for >=14 qubits when using batch-parallelization)
            options: Default options.
        """
        self._circuit = circuit
        if deactivate_numpy_multithreading is None:
            if circuit.num_qubits >= 14:
                # For 14 or more qubits the automatic parallelization of numpy methods like e.g. np.dot() in
                # Statevector-evaluation might interfere with batch-parallelization and decrease performance
                self._deactivate_numpy_multithreading = True
            else:
                self._deactivate_numpy_multithreading = False
        super().__init__(options)

    def run(
        self,
        observables: Sequence[BaseOperator] | BaseOperator,
        parameter_values: Sequence[Sequence[float]] | Sequence[float],
        **run_options,
    ) -> T:
        """Run the job of the estimation of expectation value(s).

        ``observables``, and ``parameter_values`` are used to evaluate all pairwise combinations.
        The [i, j]-th element of the result is the expectation of parameter set `i` w.r.t. observable `j`

        Args:
            observables: one or more observable objects.
            parameter_values: list of set of concrete parameters to be bound.
            run_options: runtime options used for circuit execution.

        Returns:
            The job object of EstimatorResult.

        Raises:
            TypeError: Invalid argument type given.
            ValueError: Invalid argument values given.
        """

        if isinstance(observables, BaseOperator):
            # Allow a single observable to be passed in.
            observables = (observables,)

        # allow for setups with no input parameters, i.e. cut the leading `None`
        if parameter_values[0] is None:
            parameter_values = parameter_values[1:]
        if parameter_values[0][0] is None:
            parameter_values = parameter_values[:,1:]

        if isinstance(parameter_values[0], (float, np.floating)):
            # Allow for a single set of inputs
            parameter_values = (parameter_values,)

        # Cross-validation
        self._cross_validate_circuit_parameter_values(self._circuit, parameter_values)
        self._cross_validate_circuit_observables(self._circuit, observables)

        # Options
        run_opts = copy(self.options)
        run_opts.update_options(**run_options)

        return self._run(
            observables,
            parameter_values,
            **run_opts.__dict__,
        )

    @abstractmethod
    def _run(
        self,
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> T:
        raise NotImplementedError("The subclass of BaseEstimator must implement `_run` method.")

    @staticmethod
    def _cross_validate_circuit_observables(
        circuit: QuantumCircuit, observables: Sequence[BaseOperator]
    ) -> None:
        for i, observable_ in enumerate(observables):
            if circuit.num_qubits != observable_.num_qubits:
                raise ValueError(
                    f"The number of qubits of the circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable_.num_qubits})."
                )

    @property
    def circuit(self) -> QuantumCircuit:
        """Quantum circuits that represents quantum states.

        Returns:
            The quantum circuits.
        """
        return self._circuit

    def activate_numpy_multiprocessing(self):
        """
        Activates the use of numpy-multithreading (e.g. for np.dot() in Statevector-evaluation). Will be used for
        system sizes of about >=14 qubits, might slow down batch-parallelization significantly. By default, it will
        be deactivated.
        """
        self._deactivate_numpy_multithreading = False

    def deactivate_numpy_multiprocessing(self):
        """
        Deactivates the use of numpy-multithreading (e.g. for np.dot() in Statevector-evaluation). Will be used for
        system sizes of about >=14 qubits, might slow down batch-parallelization significantly. By default, it will
        be deactivated.
        """
        self._deactivate_numpy_multithreading = True
