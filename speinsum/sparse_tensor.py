"""
Sparse tensors and operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from torch import Tensor
from speinsum.typing import Dimension


@dataclass
class SparseTensor:
    """Sparse tensor class."""

    indices: Tensor  # TODO: or None if all dimensions dense
    values: Tensor
    dimensions: Tuple[Dimension, ...]

    def default_dimension_mapping(self) -> Dict[int, int]:
        """Default dimension mapping."""
        sparse_dims = 0
        dense_dims = 1  # 0 is reserved for nnz axis
        mapping = {}
        for i, dim in enumerate(self.dimensions):
            if dim.is_sparse:
                mapping[i] = sparse_dims
                sparse_dims += 1
            else:
                mapping[i] = dense_dims
                dense_dims += 1
        return mapping

    # mapping from tensor dimension to sparse/dense dimension
    # by default, corresponding dimensions are assumed to appear in order. ie, the second dense
    # dimension is the second dimension in the values tensor.
    dimension_mapping: Dict[int, int] = None

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

        if self.dimension_mapping is None:
            self.dimension_mapping = self.default_dimension_mapping()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return tuple(dim.size for dim in self.dimensions)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the tensor."""
        return len(self.dimensions)

    @property
    def nnz(self) -> int:
        """Get the number of non-zero elements in the tensor."""
        return self.values.shape[0]

    def __str__(self):
        """String representation."""
        return f"SparseTensor(\n  indices={self.indices}, \n  values={self.values}, \n  dimensions={self.dimensions}\n)"

    def get_storage_index(self, dim: int) -> int:
        """Get the index of a dimension in the tensor.
        For a sparse dimension, this is the index of the dimension in the indices tensor.
        For a dense dimension, this is the index of the dimension in the values tensor.

        Args:
            dimension: The dimension to get the index of

        Returns:
            The index of the dimension in the corresponding representation tensor
        """
        return self.dimension_mapping[dim]

    def sparse_dimensions(self) -> Tuple[int, ...]:
        """Get the indices of all sparse dimensions in this tensor."""
        return tuple(i for i, dim in enumerate(self.dimensions) if dim.is_sparse)

    def dense_dimensions(self) -> Tuple[int, ...]:
        """Get the indices of all dense dimensions in this tensor."""
        return tuple(i for i, dim in enumerate(self.dimensions) if dim.is_dense)

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
