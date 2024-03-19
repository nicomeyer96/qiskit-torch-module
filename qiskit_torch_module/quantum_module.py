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
import torch.nn as nn
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .quantum_neural_network import QNN
from .quantum_autograd import QuantumAutograd


class QuantumModule(nn.Module):
    """ This class implements a quantum torch-module, based on a underlying quantum neural network

        args:
            circuit: Quantum circuit ansatz
            encoding_params: Parameters used for data encoding, must be present in circuit | None if the circuit does
                not use data encoding
            variational_params: Parameters used for training, must be present in circuit
            variational_params_names: Names for the trainable parameter sets (default: `variational_#`)
            variational_params_initial: Initializers for the trainable parameter sets (default: 'uniform')
                choices: constant(val=1.0) | uniform(a=0.0, b=2*pi) | normal(mean=0.0, std=1.0)
            seed_init: Generate initial parameters with fixed seed (default: None)
            observables: Observables to evaluate on the circuit (default: Pauli-Z on all qubits)
            num_threads_forward: Number of parallel threads for forward computation (default: all available threads)
            num_threads_backward: Number of parallel threads for backward computation (default: all available threads)
            encoding_gradients_flag: Whether to compute gradients w.r.t. encoding parameters
                (necessary for nested modules, default: False)
    """

    def __init__(
            self,
            circuit: QuantumCircuit,
            encoding_params: Sequence[Parameter] | ParameterVector | None,
            variational_params: Sequence[Sequence[Parameter]] | Sequence[ParameterVector] | Sequence[Parameter] | ParameterVector,
            variational_params_names: Sequence[str] | str = None,
            variational_params_initial: str | tuple[str, dict[str: float]] | Sequence[float, np.floating] |
                                        Sequence[str | tuple[str, dict[str: float]] | Sequence[float]] = 'uniform',
            seed_init: int = None,
            observables: Sequence[BaseOperator] | BaseOperator | str = 'individualZ',
            num_threads_forward: int = 0,
            num_threads_backward: int = 0,
            encoding_gradients_flag: bool = False,
    ):
        super(QuantumModule, self).__init__()
        variational_params, self._variational_params_names, variational_params_initial = (
            self._preprocess_and_validate_variational_params_sets(variational_params,
                                                                  variational_params_names,
                                                                  variational_params_initial))
        # whether to compute gradients w.r.t. input (necessary for Hybrid networks)
        self._encoding_gradients_flag = encoding_gradients_flag
        # whether to compute the gradients w.r.t. a specific parameter set, can be changed with
        # the method `set_variational_gradients_flag()` to safe compute
        # BE CAREFUL, as this might lead to unexpected behaviour if computing of gradients is de-activated for a
        # parameter set that is bound to an optimizer (as respective gradients return as `None`)
        self._variational_gradients_flag = [True for _ in variational_params]
        # set up quantum neural network
        self.qnn = QNN(
            circuit=circuit,
            encoding_params=encoding_params,
            variational_params=variational_params,
            observables=observables,
            num_threads_forward=num_threads_forward,
            num_threads_backward=num_threads_backward,
        )
        self._input_size = self.qnn.num_input_parameters()
        self._output_size = self.qnn.num_observables()
        # save torch.random state from before
        torch_random_state_ = torch.random.get_rng_state()
        if seed_init is not None:
            torch.random.manual_seed(seed_init)
        # initialize torch parameter sets
        self._initialize_parameter_sets(variational_params, self._variational_params_names, variational_params_initial)
        # re-set seed to restore previous behaviour (i.e. don't interfere with potential other (un)set seeds)
        if seed_init is not None:
            torch.random.set_rng_state(torch_random_state_)

    def forward(
            self,
            input_tensor: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ Calls into QuantumAutograd (instance of torch`s autograd functionality) to compute forward pass and
        constructs tree for backward pass.

            Args:
                input_tensor: Input to the quantum module, or None if there is no input data

            Returns:
                Result of forward pass
        """
        # Allow for scenarios where no encoding parameters are used
        if input_tensor is not None:
            # Make sure everything is realized as Tensor (dtype=torch.float32 to enhance efficiency)
            if isinstance(input_tensor, list):
                # conversion list -> np.array -> torch.tensor is faster than directly list -> torch.tensor
                input_tensor = np.array(input_tensor)
            if not torch.is_tensor(input_tensor):
                input_tensor = torch.FloatTensor(input_tensor)
            else:
                input_tensor = input_tensor.to(dtype=torch.float32)
        input_tensor = QuantumAutograd.apply(
            self.qnn, self._encoding_gradients_flag, self._variational_gradients_flag,
            input_tensor, *self._trainable_parameters
        )
        return input_tensor

    def set_variational_gradients_flag(
            self,
            variational_gradients_flag: Sequence[bool] | bool
    ) -> None:
        """ Manually set flags whether to compute the gradients w.r.t. specific parameter sets
        BE CAREFUL, as this might lead to unexpected behaviour if computing of gradients is de-activated for a
        parameter set that is bound to an optimizer (as respective gradients return as `None`)

            Args:
                variational_gradients_flag: Whether to compute gradients w.r.t. the parameter sets

            Raises:
                ValueError: Invalid (number of) flags were provided
        """
        if isinstance(variational_gradients_flag, bool):
            self._variational_gradients_flag = [variational_gradients_flag for _ in self._variational_gradients_flag]
        else:
            if len(variational_gradients_flag) != len(self._variational_gradients_flag):
                raise ValueError('Mismatch, provided {} flags for {} parameter sets.'
                                 .format(variational_gradients_flag, self._variational_gradients_flag))
            self._variational_gradients_flag = variational_gradients_flag

    def set_num_threads_forward(
            self,
            number_threads_forward: int
    ) -> None:
        """ Set number of threads for forward pass after initialization (`0` means all available).
        """
        self.qnn.set_num_threads_forward(number_threads_forward)

    def set_num_threads_backward(
            self,
            number_threads_backward: int
    ) -> None:
        """ Set number of threads for backward pass after initialization (`0` means all available).
        """
        self.qnn.set_num_threads_backward(number_threads_backward)

    @property
    def num_threads_forward(self):
        """ Return number of threads used for forward pass (`0` means all available)
        """
        return self.qnn.num_threads_forward()

    @property
    def num_threads_backward(self):
        """ Return number of threads used for backward pass (`0` means all available)
        """
        return self.qnn.num_threads_backward()

    @property
    def variational_(self) -> Sequence[nn.Parameter]:
        """ Returns a handle of the trainable parameters in the quantum module.

        Can be used to initialize a torch optimizer, e.g.:
        torchQNN = QuantumModule(...)
        opt = torch.optim.SGD(torchQNN.variational_, lr=0.1)

        This is equivalent to:
        opt = torch.optim.SGD(torchQNN.parameters(), lr=0.1)

        One can also use the class members to access the individual parameter sets (names user-defined or auto-generated):
        opt = torch.optim.SGD([{'params': qtmModel.variational_0}, {'params': qtmModel.variational_1}], lr=0.1)
        or
        opt = torch.optim.SGD([{'params': qtmModel.variational_[0]}, {'params': qtmModel.variational_[0]}], lr=0.1)
        In this case equivalent to above, but can be used to set e.g. different learning rates for parameter sets.
        A summary of all available parameter sets can also be visualized:
        print(torchQNN)
        """
        return self._trainable_parameters

    @property
    def num_trainable_parameters(self) -> int:
        """ Returns number of trainable parameters in the quantum module.
        """
        return sum([len(_tp) for _tp in self._trainable_parameters])

    @property
    def input_size(self) -> int:
        """ Returns the input size of the quantum module (i.e. number of encoding parameters)
        """
        return self._input_size

    @property
    def output_size(self) -> int:
        """ Returns the output size of the quantum module (i.e. number of observables)
        """
        return self._output_size

    @property
    def circuit(self) -> QuantumCircuit:
        """ Returns the underlying (pre-processed and alphabetically ordered) circuit
        """
        return self.qnn.circuit()

    @property
    def encoding_parameters(self) -> Sequence[Parameter]:
        """ Returns the underlying (pre-processed and alphabetically ordered) encoding parameters
        """
        return self.qnn.input_parameters()

    @property
    def variational_parameters(self) -> Sequence[Sequence[Parameter]]:
        """ Returns the underlying (pre-processed and alphabetically ordered) variational parameters
        """
        return self.qnn.trainable_parameters()

    def extra_repr(self) -> str:
        """ String-representation if the circuit. Gets returned when calling
        torchQNN = QuantumModule(...)
        print(torchQNN)
        """
        trainable_sets_metadata = []
        for _vpm, _tp in zip(self._variational_params_names, self._trainable_parameters):
            trainable_sets_metadata.append('({}) `{}`'.format(len(_tp), _vpm))
        print_sum_trainable = '' if 1 == len(trainable_sets_metadata) else '({}) '.format(self.num_trainable_parameters)
        metadata = ('input_size={}, output_size={}, num_qubits={}\n{}trainable: {}'
                    .format(self.input_size, self.output_size, self.qnn.circuit().num_qubits,
                            print_sum_trainable, trainable_sets_metadata[0]))
        for tsm in trainable_sets_metadata[1:]:
            metadata += '\n                {}'.format(tsm)
        # metadata += 'output_size={}'.format(self.output_size)
        return metadata
        # Alternative short representation
        # return f'input_size={self.input_size}, output_size={self.output_size}, trainable_parameters={self.num_trainable_parameters}'

    def _initialize_parameter_sets(
            self,
            variational_params: Sequence[Sequence[Parameter]],
            variational_params_names: Sequence[str],
            variational_params_initial: Sequence[tuple[str, dict[str: float]] | Sequence[float]],
    ) -> None:
        """ Sets up the actual trainable torch Parameters that are tracked via autograd

            Args:
                variational_params: Parameters used for training, must be present in circuit
                variational_params_names: Names for the trainable parameter sets
                variational_params_initial: Initializers for the trainable parameter sets

            Raises:
                ValueError: Wrong / inconclusive initializer instructions
        """
        self._trainable_parameters = []
        for set_index, (variational_params_, variational_params_names_, variational_params_initial_) \
                in enumerate(zip(variational_params, variational_params_names, variational_params_initial)):
            if isinstance(variational_params_initial_, tuple):
                # initialization via (method, setup dictionary)
                # choices and defaults: constant(val=1.0) | uniform(a=0.0, b=2*pi) | normal(mean=0.0, std=1.0)
                if 'constant' == variational_params_initial_[0]:
                    trainable_params_ = nn.init.constant_(torch.empty(len(variational_params_)),
                                                          val=variational_params_initial_[1].get('val', 1.0))
                elif 'uniform' == variational_params_initial_[0]:
                    trainable_params_ = nn.init.uniform_(torch.empty(len(variational_params_)),
                                                         a=variational_params_initial_[1].get('a', 0.0),
                                                         b=variational_params_initial_[1].get('b', 2*np.pi))
                elif 'normal' == variational_params_initial_[0]:
                    trainable_params_ = nn.init.normal_(torch.empty(len(variational_params_)),
                                                        mean=variational_params_initial_[1].get('mean', 0.0),
                                                        std=variational_params_initial_[1].get('std', 1.0))
                else:
                    raise ValueError('The initialization method `{}` for parameter set #{} is not available.'
                                     .format(variational_params_initial_[0], set_index))
            else:
                # initialization with explicit values
                if len(variational_params_initial_) != len(variational_params_):
                    raise ValueError('Tried to initialize Parameter set #{} of length {} with Sequence of length {}.'
                                     .format(set_index, len(variational_params_), len(variational_params_initial_)))
                trainable_params_ = torch.FloatTensor(variational_params_initial_)
            # set up and register trainable parameter container with name `variational_params_names_`
            trainable_params_ = nn.Parameter(trainable_params_)
            self.register_parameter(variational_params_names_, trainable_params_)
            # setattr(self, name, value) is equivalent to self.name = value; this allows to retrieve the respective
            # parameter sets via QuantumModule.name (i.e. for passing them to an optimizer)
            setattr(self, variational_params_names_, trainable_params_)
            # put into one collection of trainable parameter sets
            self._trainable_parameters.append(getattr(self, variational_params_names_))

    @ staticmethod
    def _preprocess_and_validate_variational_params_sets(
            variational_params: Sequence[Sequence[Parameter]] | Sequence[Parameter],
            variational_params_names: Sequence[str] | str,
            variational_params_initial: str | tuple[str, dict[str: float]] | Sequence[float, np.floating] |
                                        Sequence[str | tuple[str, dict[str: float]] | Sequence[float]]
    ) -> tuple[Sequence[Sequence[Parameter]], Sequence[str], Sequence[tuple[str, dict[str: float]] | Sequence[float]]]:
        """ Validate and pre-process parameters and associated metadata (names and initializer instructions)

            Args:
                variational_params: Parameters used for training, must be present in circuit
                variational_params_names: Names for the trainable parameter sets
                variational_params_initial: Initializers for the trainable parameter sets

            Returns:
                Cleaned-up version of parameters and associated metadata, ready to construct QNN and torch Parameters

            Raises:
                Value Error: Invalid naming or initialization instructions
        """
        if (isinstance(variational_params[0], (Parameter, ParameterExpression))
                or isinstance(variational_params, ParameterVector)):
            # singleton variational_params set
            variational_params = (variational_params, )
            # preprocess variational_params_names
            if variational_params_names is None:
                variational_params_names = ('variational', )
            elif isinstance(variational_params_names, str):
                # singleton variational_params_names
                if 'variational_' == variational_params_names:
                    raise ValueError('The name `variational_` is already occupied and can not be used for parameter set'
                                     ' naming.')
                variational_params_names = (variational_params_names, )
            else:
                raise ValueError('A singleton parameter set was given, but multiple naming instructions.')
            # preprocess variational_params_initial
            if isinstance(variational_params_initial, str):
                variational_params_initial = (variational_params_initial, {})
            if isinstance(variational_params_initial, tuple) and isinstance(variational_params_initial[1], dict):
                # singleton variational_params_initial (tuple)
                variational_params_initial = (variational_params_initial, )
            elif isinstance(variational_params_initial, Sequence) and isinstance(variational_params_initial[0], (float, np.floating)):
                # singleton variational_params_initial (explicit Sequence)
                variational_params_initial = (variational_params_initial, )
            else:
                raise ValueError('A singleton parameter set was given, but multiple initializer instructions.')
        else:
            # multiple variational_params set
            # preprocess variational_params_names
            if variational_params_names is None:
                variational_params_names = ['variational_{}'.format(set_index) for set_index in range(len(variational_params))]
            elif isinstance(variational_params_names, str):
                if 'variational_' == variational_params_names:
                    raise ValueError('The name `variational_` is already occupied and can not be used for parameter set naming.')
                variational_params_names = ['{}_{}'.format(variational_params_names, set_index) for set_index in range(len(variational_params))]
            else:
                if len(variational_params_names) != len(variational_params):
                    raise ValueError('A different number of variational_params sets ({}) and naming instructions ({}) '
                                     'were provided.'.format(len(variational_params), len(variational_params_names)))
                if 'variational_' in variational_params_names:
                    raise ValueError('The name `variational_` is already occupied and can not be used for parameter set naming.')
                variational_params_names_ = np.unique(variational_params_names)
                if len(variational_params_names_) != len(variational_params):
                    raise ValueError('The elements in variational_params_names have to be unique.')
            # preprocess variational_params_input
            if isinstance(variational_params_initial, str):
                variational_params_initial = (variational_params_initial, {})
            if isinstance(variational_params_initial, tuple) and isinstance(variational_params_initial[1], dict):
                # singleton variational_params_initial (tuple) -> copy for each variational_params set
                variational_params_initial = [variational_params_initial for _ in variational_params]
            elif isinstance(variational_params_initial, Sequence) and isinstance(variational_params_initial[0], (float, np.floating)):
                # singleton variational_params_initial (explicit Sequence) -> copy for each variational_params set
                variational_params_initial = [variational_params_initial for _ in variational_params]
            elif isinstance(variational_params_initial, np.ndarray):
                raise ValueError('Initial parameter values must be provided as a Sequence / list, not a numpy array.')
            else:
                # Sequence of variational_params_initial
                if len(variational_params_initial) != len(variational_params):
                    raise ValueError('A different number of variational_params sets ({}) and initializer instructions ({}) '
                                     'were provided.'.format(len(variational_params), len(variational_params_initial)))
                for set_index, variational_params_initial_ in enumerate(variational_params_initial):
                    if isinstance(variational_params_initial_, str):
                        variational_params_initial_ = (variational_params_initial_, {})
                    if isinstance(variational_params_initial_, tuple) and isinstance(variational_params_initial_[1], dict):
                        pass
                    elif isinstance(variational_params_initial_, Sequence) and isinstance(variational_params_initial_[0],
                                                                                          (float, np.floating)):
                        pass
                    elif isinstance(variational_params_initial_, np.ndarray):
                        raise ValueError('Initial parameter values must be provided as a Sequence / list, not a numpy array.')
                    else:
                        raise ValueError('A unknown initializer instruction was provided.')
                    variational_params_initial[set_index] = variational_params_initial_
        return variational_params, variational_params_names, variational_params_initial
