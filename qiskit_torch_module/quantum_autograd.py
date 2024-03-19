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

import torch
from torch.autograd import Function
from typing import Any
from collections.abc import Sequence

from .quantum_neural_network import QNN


class QuantumAutograd(Function):
    """ Implements torch`s autograd functionality to realize automatic differentiation of quantum neural networks.
    """

    # pylint: disable=arguments-differ
    @staticmethod
    def forward( # noqa
            ctx: Any,
            qnn: QNN,
            input_gradients_flag: bool | None,
            weights_gradients_flag: Sequence[bool],
            input_tensor: torch.Tensor,
            *weights_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """ Realizes forward pass, saves context for backward pass

            Args:
                ctx: Context
                qnn: Quantum neural network instance
                input_gradients_flag: Whether to compute gradients w.r.t. input
                weights_gradients_flag: Whether to compute gradients w.r.t. the respective variational parameter sets
                input_tensor: Input to the QNN (None if to data encoding required)
                weights_tensors: Weights of the QNN, potentially multiple sets

            Returns:
                Result of forward pass

            Raises:
                ValueError: Wrong dimensionality of input
        """
        # validate input
        if input_tensor is not None and input_tensor.shape[-1] != qnn.num_input_parameters():
            raise ValueError("The dimensionality of the input ({}) is not compatible with the input size of the "
                             "underlying quantum neural network ({}).".format(input_tensor.shape[-1],
                                                                              qnn.num_input_parameters()))
        # save for backward pass
        ctx.qnn = qnn
        ctx.input_gradients_flag = input_gradients_flag
        ctx.weights_gradients_flag = weights_gradients_flag
        ctx.save_for_backward(input_tensor, *weights_tensors)
        # compute expectation values via forward pass
        values = qnn.forward(
            input_tensor.detach().cpu().numpy() if input_tensor is not None else None,
            [weights_tensor.detach().cpu().numpy() for weights_tensor in weights_tensors],
        )
        # input was singleton -> output should also be singleton
        if input_tensor is None or 1 == len(input_tensor.shape):
            values = values[0]
        device = input_tensor.device if input_tensor is not None else 'cpu'
        return torch.FloatTensor(values).to(device)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward( # noqa
            ctx,
            grad_output: torch.Tensor,
    ) -> tuple[None, None, None, torch.Tensor, Any]:
        """ Realizes forward pass, therefore restores context for forward pass

            Args:
                ctx: Context
                grad_output: Gradient output from previous layer (or from loss function if this is the last one)

            Returns:
                Gradients w.r.t. parameter sets

            Raises:
                ValueError: Wrong dimensionality of input
        """
        qnn = ctx.qnn
        all_tensors = ctx.saved_tensors
        input_tensor = all_tensors[0]
        weights_tensors = all_tensors[1:]
        # validate input
        if input_tensor is not None and input_tensor.shape[-1] != qnn.num_input_parameters():
            raise ValueError("The dimensionality of the input ({}) is not compatible with the input size of the "
                             "underlying quantum neural network ({}).".format(input_tensor.shape[-1],
                                                                              qnn.num_input_parameters()))
        # handle singleton grad_output tensor
        if 1 == len(grad_output.shape):
            grad_output = grad_output.view(1, -1)
        grad_output_detached = grad_output.detach().cpu()
        # compute gradients
        gradients_input, gradients_weights = qnn.backward(
            input_tensor.detach().cpu().numpy() if input_tensor is not None else None,
            [weights_tensor.detach().cpu().numpy() for weights_tensor in weights_tensors],
            encoding_gradients=ctx.input_gradients_flag, variational_gradients=ctx.weights_gradients_flag
        )
        # handle gradients w.r.t. input parameters
        if gradients_input is not None:
            gradients_input = torch.FloatTensor(gradients_input)
            # account for gradients from consecutive layer, i.e. compute einsum for output `o` (i.e. o-th
            # observable) to get the overall gradient w.r.t. parameter `p` and batch 'b'
            gradients_input = torch.einsum(
                'bo,bop->bp', grad_output_detached, gradients_input
            )
            gradients_input = gradients_input.to(input_tensor.device)
        # temporary buffer for returning gradients w.r.t. weight parameters
        all_gradients_weights = []
        # handle gradients w.r.t. weight parameters
        for set_index, gradients_weights_ in enumerate(gradients_weights):
            if gradients_weights_ is None:
                all_gradients_weights.append(None)
                continue
            # handle gradients w.r.t. weight parameters
            gradients_weights_ = torch.FloatTensor(gradients_weights_)
            # account for gradients from consecutive layer, i.e. compute einsum for batch `b` and output `o` (i.e. j-th
            # observable) to get the overall gradient w.r.t. parameter `p`
            gradients_weights_ = torch.einsum(
                'bo,bop->p', grad_output_detached, gradients_weights_
            )
            gradients_weights_ = gradients_weights_.to(weights_tensors[set_index].device)
            all_gradients_weights.append(gradients_weights_)
        return None, None, None, gradients_input, *all_gradients_weights
