"""
Sparse tensors and operations.
"""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum
from torch import Tensor


class DimensionFormat(Enum):
    """Dimension format."""

    SPARSE = "s"
    DENSE = "d"


@dataclass
class SparseTensor:
    """Sparse tensor class."""

    indices: Tensor
    values: Tensor
    shape: Tuple[Tuple, ...]
    dimension_format: Tuple[DimensionFormat, ...]

    def __post_init__(self):
        """Post-initialization checks."""
        if self.indices.ndim != 2:
            raise ValueError("Indices must be a 2D tensor.")
        if self.indices.shape[0] != self.values.shape[0]:
            raise ValueError("Indices and values must have the same number of rows.")
        if self.indices.shape[1] != len(self.sparse_dimensions()):
            raise ValueError(
                "Indices must have the same number of columns as the number of sparse dimensions in the shape."
            )
        if len(self.values.shape) != len(self.dense_dimensions()) + 1:
            raise ValueError(
                "Values must have the same number of dimensions as the number of dense dimensions in the shape."
            )
        if len(self.dimension_format) != len(self.shape):
            raise ValueError(
                "Format must have the same number of dimensions as the shape."
            )

    def __str__(self):
        """String representation."""
        return f"SparseTensor(indices={self.indices}, values={self.values}, shape={self.shape}, format={self.dimension_format})"

    def sparse_dimensions(self) -> Tuple[int, ...]:
        """Get the indices of all sparse dimensions in this tensor."""
        return tuple(
            i
            for i, f in enumerate(self.dimension_format)
            if f == DimensionFormat.SPARSE
        )

    def dense_dimensions(self) -> Tuple[int, ...]:
        """Get the indices of all dense dimensions in this tensor."""
        return tuple(
            i for i, f in enumerate(self.dimension_format) if f == DimensionFormat.DENSE
        )

    def tensordot(self, other: SparseTensor, dims: Tuple[int, ...]) -> SparseTensor:
        """Compute a tensor dot product with another tensor.

        Args:
            other: The other tensor
            dims: The dimensions to dot
        """
        ...

    def transpose(self, dims: Tuple[int, ...]) -> SparseTensor:
        """Transpose the tensor.

        Args:
            dims: The dimensions to transpose
        """
        ...
