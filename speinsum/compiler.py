"""
Core compiler implementation for sparse einsum operations.
"""

from typing import List, Tuple
import torch
from tiql import intersect
from .indirecteinsum import einsum_gs
from .sparse_tensor import SparseTensor
from .typing import DimensionFormat, Dimension


# def einsum_gs(expression, **tensors):
#     print(f"Issued: {expression}")
#     print("Supplied tensors:")
#     for name, tensor in tensors.items():
#         print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
#         print(f"    {tensor}")
#     return torch.zeros((1,))


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
    a_eqn: str, b_eqn: str, out_eqn: str, a_tensor: SparseTensor, b_tensor: SparseTensor, out_format: str
) -> SparseTensor:
    """Execute a two-operand einsum operation.

    Args:
        a_eqn: Einsum equation for the first tensor
        b_eqn: Einsum equation for the second tensor
    """
    a_eqn, a_tensor = _coalesce_einsum_indices(a_eqn, a_tensor)
    b_eqn, b_tensor = _coalesce_einsum_indices(b_eqn, b_tensor)

    a_dense_dims = {
        a_eqn[d]: a_tensor.get_storage_index(d) for d in range(a_tensor.ndim) if a_tensor.dimensions[d].is_dense
    }
    a_sparse_dims = {
        a_eqn[d]: a_tensor.get_storage_index(d) for d in range(a_tensor.ndim) if a_tensor.dimensions[d].is_sparse
    }
    b_dense_dims = {
        b_eqn[d]: b_tensor.get_storage_index(d) for d in range(b_tensor.ndim) if b_tensor.dimensions[d].is_dense
    }
    b_sparse_dims = {
        b_eqn[d]: b_tensor.get_storage_index(d) for d in range(b_tensor.ndim) if b_tensor.dimensions[d].is_sparse
    }

    # Sparse Intersection
    shared_sparse = a_sparse_dims.keys() & b_sparse_dims.keys()
    a_mixed_dims = a_sparse_dims.keys() & b_dense_dims.keys()  # dimensions that are sparse in a and dense in b
    b_mixed_dims = b_sparse_dims.keys() & a_dense_dims.keys()  # dimensions that are sparse in b and dense in a

    out_sparse = set(out_eqn[i] for i in range(len(out_eqn)) if out_format[i] == "s")
    out_dense = set(out_eqn[i] for i in range(len(out_eqn)) if out_format[i] == "d")

    tensors = {}
    nnz_index = "p"  # TODO: make sure nnz_index does not appear in einsum

    if shared_sparse:
        int_idx = intersect(
            "A_shared_sp[i, c] == B_shared_sp[j, c] -> (i,j)",
            A_shared_sp=a_tensor.indices[:, [a_sparse_dims[d] for d in shared_sparse]],
            B_shared_sp=b_tensor.indices[:, [b_sparse_dims[d] for d in shared_sparse]],
        ).T
        # print("sparse intersection", int_idx)

        tensors["A_nnz"] = int_idx[:, 0]
        tensors["B_nnz"] = int_idx[:, 1]
        a_nnz = f"A_nnz[{nnz_index}]"
        b_nnz = f"B_nnz[{nnz_index}]"

        a_sparse_order = list(a_sparse_dims.keys())
        b_sparse_order = list(b_sparse_dims.keys() - a_sparse_dims.keys())
        out_sparse_order = a_sparse_order + b_sparse_order
        out_indices = torch.cat(
            [
                a_tensor.indices[int_idx[:, 0]][:, [a_sparse_dims[d] for d in a_sparse_order]],
                b_tensor.indices[int_idx[:, 1]][:, [b_sparse_dims[d] for d in b_sparse_order]],
            ],
            dim=1,
        )

    elif a_mixed_dims and not b_mixed_dims:
        # All sparse dimensions in a_tensor
        tensors["B_nnz"] = torch.zeros((a_tensor.nnz))
        a_nnz = nnz_index
        b_nnz = f"B_nnz[{nnz_index}]"

        out_sparse_order = list(a_sparse_dims)
        out_indices = a_tensor.indices[:, [a_sparse_dims[d] for d in out_sparse_order]]

    elif b_mixed_dims and not a_mixed_dims:
        # All sparse dimensions in b_tensor
        tensors["A_nnz"] = torch.zeros((b_tensor.nnz,))
        a_nnz = f"A_nnz[{nnz_index}]"
        b_nnz = nnz_index

        out_sparse_order = list(b_sparse_dims)
        out_indices = b_tensor.indices[:, [b_sparse_dims[d] for d in out_sparse_order]]

    else:
        int_idx = torch.stack(
            [
                torch.arange(a_tensor.nnz).repeat_interleave(b_tensor.nnz),
                torch.arange(b_tensor.nnz).expand(a_tensor.nnz),
            ]
        )
        # print("cross product", int_idx)

        tensors["A_nnz"] = int_idx[:, 0]
        tensors["B_nnz"] = int_idx[:, 1]
        a_nnz = f"A_nnz[{nnz_index}]"
        b_nnz = f"B_nnz[{nnz_index}]"

        a_sparse_order = list(a_sparse_dims.keys())
        b_sparse_order = list(b_sparse_dims.keys() - a_sparse_dims.keys())
        out_sparse_order = a_sparse_order + b_sparse_order
        out_indices = torch.cat(
            [
                a_tensor.indices[int_idx[:, 0]][:, [a_sparse_dims[d] for d in a_sparse_order]],
                b_tensor.indices[int_idx[:, 1]][:, [b_sparse_dims[d] for d in b_sparse_order]],
            ],
            dim=1,
        )

    # Gather Einsum
    tensors["A_crd"] = a_tensor.indices
    tensors["B_crd"] = b_tensor.indices

    tensors["A_val"] = a_tensor.values
    tensors["B_val"] = b_tensor.values

    out_dims = set(out_eqn)
    sparse_dims = a_sparse_dims.keys() | b_sparse_dims.keys()
    dense_dims = list(a_dense_dims.keys() & b_dense_dims.keys())
    dense_sizes = [out_indices.shape[0]]
    for idx in dense_dims:
        if idx in a_dense_dims:
            dense_sizes.append(a_tensor.values.shape[a_dense_dims[idx]])
        else:
            dense_sizes.append(b_tensor.values.shape[b_dense_dims[idx]])

    # sparse -> dense conversion
    exclude = set()
    for dim in sparse_dims - out_sparse:
        i = out_sparse_order.index(dim)
        if dim in out_dense:
            gather_tensor_name = f"{dim}_G"
            tensors[gather_tensor_name] = out_indices[:, i]
            dense_dims.append(f"{gather_tensor_name}[{nnz_index}]")
            if dim in a_sparse_dims:
                dense_sizes.append(a_tensor.shape[a_eqn.index(dim)])
            else:
                dense_sizes.append(b_tensor.shape[b_eqn.index(dim)])

        exclude.add(i)

    out_sparse_order = [out_sparse_order[i] for i in range(len(out_sparse_order)) if i not in exclude]
    out_indices = out_indices[:, [i for i in range(out_indices.shape[1]) if i not in exclude]]

    # einsum_gs construction
    tensors["Out_val"] = torch.zeros(dense_sizes)
    eqn_lhs = f"Out_val[{", ".join([nnz_index] + dense_dims)}]"
    eqn = eqn_lhs + " += "

    a_dense_index = [None] * len(a_tensor.values.shape)
    a_dense_index[0] = a_nnz
    for dim, i in a_dense_dims.items():
        if dim in b_sparse_dims:
            a_dense_index[i] = f"B_crd[{b_nnz}, {b_sparse_dims[dim]}]"
        else:
            a_dense_index[i] = dim
    eqn += f"A_val[{",".join(a_dense_index)}]"
    eqn += " * "

    b_dense_index = [None] * len(b_tensor.values.shape)
    b_dense_index[0] = b_nnz
    for dim, i in b_dense_dims.items():
        if dim in a_sparse_dims:
            b_dense_index[i] = f"A_crd[{a_nnz}, {a_sparse_dims[dim]}]"
        else:
            b_dense_index[i] = dim
    eqn += f"B_val[{",".join(b_dense_index)}]"

    print(eqn)
    out_val = einsum_gs(eqn, **tensors)
    print("intermediate val:")
    print(out_val)
    print("intermediate indices:")
    print(out_indices)

    # Sparse Coalescing
    if len(sparse_dims - out_dims) > 0:
        # perform sparse coalescing if there is a reduction over some sparse dimension.
        if out_indices.shape[1] > 0:
            unique_indices, gather_idx = torch.unique(out_indices, dim=0, return_inverse=True)
        else:
            unique_indices, gather_idx = torch.zeros((1, 0)), torch.zeros((out_indices.shape[0],), dtype=torch.long)
        coalesce_eqn = f"Out[{",".join([f"G[{nnz_index}]"] + dense_dims)}] += {eqn_lhs}"
        dense_sizes[0] = unique_indices.shape[0]
        out_val = einsum_gs(coalesce_eqn, Out_val=out_val, G=gather_idx, Out=torch.zeros(dense_sizes))
        out_indices = unique_indices

    dimension_format = []
    dimension_mapping = {}
    for i, idx in enumerate(out_eqn):
        if idx in dense_dims:
            dimension_mapping[i] = dense_dims.index(idx) + 1
            dimension_format.append(
                Dimension(
                    size=dense_sizes[dense_dims.index(idx) + 1],
                    format=DimensionFormat.DENSE,
                )
            )
        else:
            dimension_mapping[i] = out_sparse_order.index(idx)
            if idx in a_sparse_dims:
                size = a_tensor.shape[a_eqn.index(idx)]
            else:
                size = b_tensor.shape[b_eqn.index(dim)]
            dimension_format.append(Dimension(size=size, format=DimensionFormat.SPARSE))

    return SparseTensor(out_indices, out_val, dimension_format, dimension_mapping)


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
