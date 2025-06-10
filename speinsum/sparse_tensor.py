"""
Sparse tensors and operations.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Tuple, Sequence
import torch
from torch import Tensor
from speinsum.typing import Dimension, DimensionFormat


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

    @staticmethod
    def random_sparse_tensor(dims: Sequence[Dimension], approx_nonzeros: int):
        """
        Generates a random sparse tensor given a list of Dimension objects.

        Sparse dimensions contribute to the index set.
        Dense dimensions contribute to the shape of each stored value.

        Args:
            dims (List[Dimension]): List of dimensions, each with size and format (Sparse or Dense).
            approx_nonzeros (int): Approximate number of non-zero entries to generate.

        Returns:
            SparseTensor: A sparse tensor with generated indices and random dense values.
        """
        sparse_dims = [d for d in dims if d.format == DimensionFormat.SPARSE]
        dense_dims = [d for d in dims if d.format == DimensionFormat.DENSE]

        sparse_sizes = [d.size for d in sparse_dims]
        dense_sizes = [d.size for d in dense_dims]

        total_sparse_space = math.prod(sparse_sizes)

        # Estimate number of samples needed to reduce collision
        if total_sparse_space - approx_nonzeros > 0.1 * total_sparse_space:
            sample_count = max(
                int(total_sparse_space * math.log(total_sparse_space / (total_sparse_space - approx_nonzeros))),
                approx_nonzeros,
            )
        else:
            sample_count = int(total_sparse_space * math.log(total_sparse_space))

        # Generate sparse indices with shape (num_sparse_dims, actual_nnz)
        index_tensors = [torch.randint(0, size, (sample_count,)) for size in sparse_sizes]
        raw_indices = torch.stack(index_tensors, dim=1)
        indices = torch.unique(raw_indices, dim=0)

        nnz = indices.size(0)

        # Generate values with shape (nnz, *dense_sizes)
        values_shape = (nnz, *dense_sizes)
        values = torch.randn(values_shape)

        # Build mapping: maps from original dim index → storage index
        mapping: Dict[int, int] = {}
        sparse_counter = 0
        dense_counter = 1  # because dim 0 in values is for nnz

        for i, d in enumerate(dims):
            if d.format == DimensionFormat.SPARSE:
                mapping[i] = sparse_counter
                sparse_counter += 1
            else:
                mapping[i] = dense_counter
                dense_counter += 1

        return SparseTensor(indices=indices, values=values, dimensions=dims, dimension_mapping=mapping)

    @staticmethod
    def from_dense(self, tensor: Tensor, formats: Sequence[DimensionFormat], sparse_rate: float = 0.8) -> SparseTensor:
        """Generate a new sparse tensor given a dense tensor and the format of each dimension.

        Args:
            tensor (Tensor): the tensor from which data is derived.
            formats (Sequence[DimensionFormat]): the desired format of each dimension in the dense tensor.
            Must have the same length as tensor.shape
            sparse_rate(float): a value in the range [0,1] that determines how much sparsity to add to each sparse
            dimension. Eg. if 0.5 is given, half of the indices from the original tensor will be keep in that sparse
            dimension.

        Returns:
            SparseTensor: a new sparse tensor derived from the given dense tensor
        """
        ...

    def to_dense(self) -> Tensor:
        """
        Converts the sparse tensor to its dense representation.

        Returns:
            Tensor: A dense tensor of shape `self.shape` where the sparse indices
            are populated with corresponding values, and all other entries are zero.
        """
        dense = torch.zeros(self.shape)

        for i in range(self.indices.shape[0]):
            index = tuple(
                (
                    slice(None)
                    if self.dimensions[j].format == DimensionFormat.DENSE
                    else self.indices[i, self.get_storage_index(j)]
                )
                for j in range(len(self.dimensions))
            )
            dense[index] = self.values[i]

        return dense

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
