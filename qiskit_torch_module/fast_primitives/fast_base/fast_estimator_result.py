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

##############################################################################################################
# This code is a modified version of qiskit.primitives.base.estimator_result and part of Qiskit-Torch-Module #
# The main modifications include:                                                                            #
# - An alternated data structure to be compatible with the other modifications                               #
##############################################################################################################

"""
FastEstimatorResult class
"""

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class FastEstimatorResult:
    """Result of FastEstimator."""

    values: Sequence[Sequence[float]]
    """The expectation values."""
    metadata: Sequence[Sequence[dict[str, Any]]]
    """Additional information about the job."""
