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
from qiskit.circuit import Parameter, ParameterVector

from . import QuantumModule


class HybridModule(nn.Module):
    """ This class implements a hybrid torch-module, by embedding a quantum module (with Pauli-Z observables on all
    individual qubits) between two linear layers.

        args:
            circuit: Quantum circuit ansatz
            encoding_params: Parameters used for data encoding, must be present in circuit | None if the circuit does
                not use data encoding
            variational_params: Parameters used for training, must be present in circuit
            input_size: Dimensionality of input (does NOT need to correspond to number of encoding parameters)
            output_size: Dimensionality of output (does NOT need to correspond to number of observables /qubits)
            variational_params_names: Names for the trainable parameter sets (default: `variational_#`)
            variational_params_initial: Initializers for the trainable parameter sets (default: 'uniform')
                choices: constant(val=1.0) | uniform(a=0.0, b=2*pi) | normal(mean=0.0, std=1.0)
            seed_init: Generate initial parameters with fixed seed (default: None)
            num_threads_forward: Number of parallel threads for forward computation (default: all available threads)
            num_threads_backward: Number of parallel threads for backward computation (default: all available threads)
    """

    def __init__(
            self,
            circuit: QuantumCircuit,
            encoding_params: Sequence[Parameter] | ParameterVector | None,
            variational_params: Sequence[Sequence[Parameter]] | Sequence[ParameterVector] | Sequence[Parameter] | ParameterVector,
            input_size: int,
            output_size: int,
            variational_params_names: Sequence[str] | str = None,
            variational_params_initial: str | tuple[str, dict[str: float]] | Sequence[float, np.floating] |
                                        Sequence[str | tuple[str, dict[str: float]] | Sequence[float]] = 'uniform',
            seed_init: int = None,
            num_threads_forward: int = 0,
            num_threads_backward: int = 0,
    ):
        super(HybridModule, self).__init__()
        # save torch.random state from before
        torch_random_state_ = torch.random.get_rng_state()
        if seed_init is not None:
            torch.random.manual_seed(seed_init)
        # initialize classical preprocessing layer (keep this order to ensure printing in right order)
        self._preprocessing = nn.Linear(in_features=input_size, out_features=len(encoding_params), bias=True)
        self._input_size = input_size
        # initialize quantum module
        # - single-qubit observables have to be evaluated on all qubits, as this is assumed by the postprocessing NN
        # - gradients w.r.t. input parameters have to be computed in order to propagate them to the preprocessing NN
        # - the seed is set to None, as a `global` seed can also be set in the hybrid module
        self._quantum_module = QuantumModule(
            circuit=circuit,
            encoding_params=encoding_params,
            variational_params=variational_params,
            variational_params_names=variational_params_names,
            variational_params_initial=variational_params_initial,
            observables='individualZ',
            num_threads_forward=num_threads_forward,
            num_threads_backward=num_threads_backward,
            encoding_gradients_flag=True,
            seed_init=None
        )
        # initialize classical postprocessing layer
        self._postprocessing = nn.Linear(in_features=self._quantum_module.output_size, out_features=output_size, bias=True)
        self._output_size = output_size
        # re-set seed to restore previous behaviour (i.e. don't interfere with potential other (un)set seeds)
        if seed_init is not None:
            torch.random.set_rng_state(torch_random_state_)

    def forward(
            self,
            input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass through preprocessing linear layer, quantum module, and post-processing linear layer

            Args:
                input_tensor: Input to the quantum module

            Returns:
                Result of forward pass
        """
        # Make sure everything is realized as Tensor (dtype=torch.float32 to enhance efficiency)
        if not torch.is_tensor(input_tensor):
            if isinstance(input_tensor, list):
                # conversion list -> np.array -> torch.tensor is faster than directly list -> torch.tensor
                input_tensor = np.array(input_tensor)
            input_tensor = torch.FloatTensor(input_tensor)
        else:
            input_tensor = input_tensor.to(dtype=torch.float32)
        input_tensor = self._preprocessing(input_tensor)
        input_tensor = self._quantum_module(input_tensor)
        input_tensor = self._postprocessing(input_tensor)
        return input_tensor

    @property
    def quantum_module(self):
        """ Returns underlying quantum module
        """
        return self._quantum_module

    @property
    def pre_parameters_(self):
        """ Returns a handle of the trainable parameters in the preprocessing NN (weights + bias)
        """
        return self._preprocessing.parameters()

    @property
    def quantum_parameters_(self):
        """ Returns a handle of the trainable parameters in the quantum module
        (only for convenience, can also be accessed via the methods of the `quantum_module`)

        Can be used to initialize a torch optimizer, e.g.:
        torchHQNN = HybridModule(...)
        opt = torch.optim.SGD([{'params': torchHQNN.pre_parameters}, {'params': torchHQNN.quantum_parameters_},
            {'params': torchHQNN.post_parameters_}], lr=0.1)

        This is equivalent to:
        opt = torch.optim.SGD(torchHQNN.parameters(), lr=0.1)

        One can also use the class members to access the individual parameter sets of the quantum module
        (omitting the classical parameters for now, which can be handled in a similar fashion)
        opt = torch.optim.SGD([{'params': qtmModel.quantum_module.variational_0},
            {'params': qtmModel.quantum_module.variational_1}], lr=0.1)
        or
        opt = torch.optim.SGD([{'params': qtmModel.quantum_parameters_[0]},
            {'params': qtmModel.quantum_parameters_[0]}], lr=0.1)
        In this case equivalent to above, but can be used to set e.g. different learning rates for parameter sets.
        A summary of all available parameter sets can also be visualized:
        print(torchHQNN)
        """
        return self._quantum_module.variational_

    @property
    def post_parameters_(self):
        """ Returns a handle of the trainable parameters in the preprocessing NN  (weights + bias)
        """
        return self._postprocessing.parameters()

    @property
    def num_trainable_parameters_quantum(self):
        """ Returns number of trainable parameters in the quantum part if the hybrid module.
        """
        return self._quantum_module.num_trainable_parameters

    @property
    def num_trainable_parameters_classical(self):
        """ Returns number of trainable parameters in the quantum part if the hybrid module.
        """
        pre_parameters = filter(lambda p: p.requires_grad, self._preprocessing.parameters())
        post_parameters = filter(lambda p: p.requires_grad, self._postprocessing.parameters())
        return sum([np.prod(p.size()) for p in pre_parameters]) + sum([np.prod(p.size()) for p in post_parameters])

    @property
    def num_trainable_parameters(self):
        """ Returns number of trainable parameters in the hybrid module
        """
        return self.num_trainable_parameters_quantum + self.num_trainable_parameters_classical

    @property
    def input_size(self) -> int:
        """ Returns the input size of the hybrid module
        """
        return self._input_size

    @property
    def output_size(self) -> int:
        """ Returns the output size of the hybrid module
        """
        return self._output_size
