"""
Core compiler implementation for sparse einsum operations.
"""

from typing import List, Tuple
import torch
from .sparse_tensor import SparseTensor
from .typing import DimensionFormat, Dimension


def parse_einsum_equation(equation: str) -> Tuple[List[str], str]:
    """Parse an einsum equation into input subscripts and output subscript.

    Args:
        equation: String of the form "ij,jk->ik"

    Returns:
        Tuple of (input_subscripts, output_subscript)
        where input_subscripts is a list of strings for each input tensor
        and output_subscript is a string for the output tensor
    """
    equation_sides = equation.split("->")

    inputs = equation_sides[0]
    input_subscripts = [s.strip() for s in inputs.split(",")]

    if len(equation_sides) > 2:
        raise ValueError("Invalid einsum equation.")

    if len(equation_sides) == 1:
        # Implicit output. The output indices are all non-repeated indices in the input, sorted alphabetically.
        included_indices = set()
        excluded_indices = set()

        for index in "".join(input_subscripts):
            if index in included_indices:
                excluded_indices.add(index)
            else:
                included_indices.add(index)

        output = "".join(sorted(included_indices - excluded_indices))
    else:
        # Explicit output.
        output = equation_sides[1]

    if not output:
        raise ValueError("Invalid einsum equation.")

    return input_subscripts, output.strip()


def _select_along_dim(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Select along a dimension of a tensor such that the output tensor has exactly one
    fewer dimensions than the input tensor, removing the dimension dim.

    Values are selected with respect to the first dimension of the input tensor.

    The output tensor is obtained by
    output[i_0, ..., i_{dim-1}, i_{dim+1}, ..., i_{n-1}]
       = input[i_0, ..., i_{dim-1}, index[i_0], i_{dim+1}, ..., i_{n-1}]

    Args:
        tensor: Input tensor
        dim: Dimension to select along
        index: Indices to select

    Returns:
        Output tensor
    """
    selected = torch.index_select(tensor, dim=dim, index=index)
    diag = torch.diagonal(selected, dim1=0, dim2=dim)  # torch diagonal places diagonal along final dimension
    perm = [diag.ndim - 1] + list(range(0, diag.ndim - 1))  # Rotate back to original order
    return diag.permute(perm)


def _coalesce_einsum_indices(input_eqn: str, tensor: SparseTensor) -> Tuple[str, SparseTensor]:
    """Coalesce repeated indices of a tensor input in some einsum equation
    into an equivalent einsum equation and sparse tensor that has no repeated indices.

    Args:
        input_eqn: Einsum equation for the input tensor
        tensor: Input tensor

    Returns:
        Tuple of (einsum equation, coalesced tensor)
    """
    dense_dims = [d for d in range(tensor.ndim) if tensor.dimensions[d].is_dense]
    sparse_dims = [d for d in range(tensor.ndim) if tensor.dimensions[d].is_sparse]

    # Diagonalize repeated indices in dense dimensions using einsum to combine them
    # Map dense dimensions to their corresponding labels from input equation
    value_labels = ["A"] * tensor.values.ndim  # A for nnz axis. TODO: throw error if A is in input_eqn
    for d in dense_dims:
        value_labels[tensor.get_storage_index(d)] = input_eqn[d]
    # Get unique labels to remove duplicates
    dense_indices = ["A"] + list(set(value_labels[1:]))  # Keep A as first index
    new_values = tensor.values
    if len(dense_indices) < len(value_labels):
        # Build einsum equation to diagonalize repeated indices
        diagonalize_einsum = f"{''.join(value_labels)}->{''.join(dense_indices)}"
        # Apply einsum to get values with diagonalized dimensions
        new_values = torch.einsum(diagonalize_einsum, new_values)

    # Mark repeated sparse dimensions to coalesce them
    sparse_coalesce_indices = {}
    for d in sparse_dims:
        index = input_eqn[d]
        if index not in sparse_coalesce_indices:
            sparse_coalesce_indices[index] = []
        sparse_coalesce_indices[index].append(tensor.get_storage_index(d))

    # Perform diagonalization for all unique indices in the einsum equation
    unique_indices = list(set(input_eqn))
    keep_sparse_indices = []  # all indices kept in the output sparse indices tensor

    new_indices = tensor.indices
    new_dimensions = []
    new_mapping = {}

    for j, index in enumerate(unique_indices):
        if index in sparse_coalesce_indices:
            # Sparse index
            coalesce = sparse_coalesce_indices[index]
            indices_dim = coalesce[0]
            # Diagonalize sparse dimensions by masking only coordinates with all positions equal.
            if len(coalesce) > 1:
                diagonal_mask = torch.all(new_indices[:, coalesce[1:]] == new_indices[:, [indices_dim]], dim=1)
                new_indices = new_indices[diagonal_mask, :]
                new_values = new_values[diagonal_mask]

            # Diagonalize sparse with dense dimensions by gathering at the coordinate of the sparse dimension.
            if index in dense_indices:
                remove_at = dense_indices.index(index)
                dense_index = new_indices[:, indices_dim]

                new_values = _select_along_dim(new_values, remove_at, dense_index)
                del dense_indices[remove_at]

            new_dimensions.append(Dimension(size=tensor.dimensions[indices_dim].size, format=DimensionFormat.SPARSE))
            new_mapping[j] = len(keep_sparse_indices)
            keep_sparse_indices.append(indices_dim)

        else:
            # Dense index
            values_dim = dense_indices.index(index)
            new_dimensions.append(Dimension(size=new_values.shape[values_dim], format=DimensionFormat.DENSE))
            new_mapping[j] = values_dim

    new_indices = new_indices[:, keep_sparse_indices]  # Remove redundant sparse dimensions
    new_eqn = "".join(unique_indices)  # Simplified einsum equation with no repeated indices

    return new_eqn, SparseTensor(
        indices=new_indices, values=new_values, dimensions=new_dimensions, dimension_mapping=new_mapping
    )


def _two_operand_einsum(
    a_eqn: str, b_eqn: str, out_eqn: str, a_tensor: SparseTensor, b_tensor: SparseTensor
) -> SparseTensor:
    """Execute a two-operand einsum operation.

    Args:
        a_eqn: Einsum equation for the first tensor
        b_eqn: Einsum equation for the second tensor
    """


def sparse_einsum(equation: str, out_format: str, *tensors: SparseTensor) -> SparseTensor:
    """Execute a sparse einsum operation.

    Args:
        equation: Einsum equation in the form "ij,jk->ik"
        out_format: Format of each dimension of the output tensor
        *tensors: Input tensors (mix of sparse and dense)

    Returns:
        Output tensor
    """
    # TODO: Implement actual compilation logic

    return torch.einsum(equation, *tensors)
