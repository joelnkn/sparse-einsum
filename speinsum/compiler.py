"""
Core compiler implementation for sparse einsum operations.
"""

from typing import List, Tuple
from collections import defaultdict
import torch
from tiql import intersect, table_intersect
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


# # def _two_operand_einsum(
#     a_eqn: str, b_eqn: str, out_eqn: str, a_tensor: SparseTensor, b_tensor: SparseTensor, out_format: str
# ) -> SparseTensor:
#     """Execute a two-operand einsum operation.

#     Args:
#         a_eqn: Einsum equation for the first tensor
#         b_eqn: Einsum equation for the second tensor
#     """
#     a_eqn, a_tensor = _coalesce_einsum_indices(a_eqn, a_tensor)
#     b_eqn, b_tensor = _coalesce_einsum_indices(b_eqn, b_tensor)

#     a_dense_dims = {
#         a_eqn[d]: a_tensor.get_storage_index(d) for d in range(a_tensor.ndim) if a_tensor.dimensions[d].is_dense
#     }
#     a_sparse_dims = {
#         a_eqn[d]: a_tensor.get_storage_index(d) for d in range(a_tensor.ndim) if a_tensor.dimensions[d].is_sparse
#     }
#     b_dense_dims = {
#         b_eqn[d]: b_tensor.get_storage_index(d) for d in range(b_tensor.ndim) if b_tensor.dimensions[d].is_dense
#     }
#     b_sparse_dims = {
#         b_eqn[d]: b_tensor.get_storage_index(d) for d in range(b_tensor.ndim) if b_tensor.dimensions[d].is_sparse
#     }

#     # Sparse Intersection
#     shared_sparse = a_sparse_dims.keys() & b_sparse_dims.keys()
#     a_mixed_dims = a_sparse_dims.keys() & b_dense_dims.keys()  # dimensions that are sparse in a and dense in b
#     b_mixed_dims = b_sparse_dims.keys() & a_dense_dims.keys()  # dimensions that are sparse in b and dense in a

#     out_sparse = set(out_eqn[i] for i in range(len(out_eqn)) if out_format[i] == "s")
#     out_dense = set(out_eqn[i] for i in range(len(out_eqn)) if out_format[i] == "d")

#     tensors = {}
#     nnz_index = "p"  # TODO: make sure nnz_index does not appear in einsum

#     if shared_sparse:
#         int_idx = intersect(
#             "A_shared_sp[i, c] == B_shared_sp[j, c] -> (i,j)",
#             A_shared_sp=a_tensor.indices[:, [a_sparse_dims[d] for d in shared_sparse]],
#             B_shared_sp=b_tensor.indices[:, [b_sparse_dims[d] for d in shared_sparse]],
#         ).T
#         # print("sparse intersection", int_idx)

#         tensors["A_nnz"] = int_idx[:, 0]
#         tensors["B_nnz"] = int_idx[:, 1]
#         a_nnz = f"A_nnz[{nnz_index}]"
#         b_nnz = f"B_nnz[{nnz_index}]"

#         a_sparse_order = list(a_sparse_dims.keys())
#         b_sparse_order = list(set(b_sparse_dims.keys()) - set(a_sparse_dims.keys()))
#         out_sparse_order = a_sparse_order + b_sparse_order
#         out_indices = torch.cat(
#             [
#                 a_tensor.indices[int_idx[:, 0]][:, [a_sparse_dims[d] for d in a_sparse_order]],
#                 b_tensor.indices[int_idx[:, 1]][:, [b_sparse_dims[d] for d in b_sparse_order]],
#             ],
#             dim=1,
#         )

#     elif a_mixed_dims and not b_mixed_dims:
#         # All sparse dimensions in a_tensor
#         tensors["B_nnz"] = torch.zeros((a_tensor.nnz))
#         a_nnz = nnz_index
#         b_nnz = f"B_nnz[{nnz_index}]"

#         out_sparse_order = list(a_sparse_dims)
#         out_indices = a_tensor.indices[:, [a_sparse_dims[d] for d in out_sparse_order]]

#     elif b_mixed_dims and not a_mixed_dims:
#         # All sparse dimensions in b_tensor
#         tensors["A_nnz"] = torch.zeros((b_tensor.nnz,))
#         a_nnz = f"A_nnz[{nnz_index}]"
#         b_nnz = nnz_index

#         out_sparse_order = list(b_sparse_dims)
#         out_indices = b_tensor.indices[:, [b_sparse_dims[d] for d in out_sparse_order]]

#     else:
#         int_idx = torch.stack(
#             [
#                 torch.arange(a_tensor.nnz).repeat_interleave(b_tensor.nnz),
#                 torch.arange(b_tensor.nnz).expand(a_tensor.nnz),
#             ]
#         )
#         # print("cross product", int_idx)

#         tensors["A_nnz"] = int_idx[:, 0]
#         tensors["B_nnz"] = int_idx[:, 1]
#         a_nnz = f"A_nnz[{nnz_index}]"
#         b_nnz = f"B_nnz[{nnz_index}]"

#         a_sparse_order = list(a_sparse_dims.keys())
#         b_sparse_order = list(b_sparse_dims.keys() - a_sparse_dims.keys())
#         out_sparse_order = a_sparse_order + b_sparse_order
#         out_indices = torch.cat(
#             [
#                 a_tensor.indices[int_idx[:, 0]][:, [a_sparse_dims[d] for d in a_sparse_order]],
#                 b_tensor.indices[int_idx[:, 1]][:, [b_sparse_dims[d] for d in b_sparse_order]],
#             ],
#             dim=1,
#         )

#     # Gather Einsum
#     tensors["A_crd"] = a_tensor.indices
#     tensors["B_crd"] = b_tensor.indices

#     tensors["A_val"] = a_tensor.values
#     tensors["B_val"] = b_tensor.values

#     out_dims = set(out_eqn)
#     sparse_dims = a_sparse_dims.keys() | b_sparse_dims.keys()
#     dense_dims = list(a_dense_dims.keys() & b_dense_dims.keys())
#     dense_sizes = [out_indices.shape[0]]
#     for idx in dense_dims:
#         if idx in a_dense_dims:
#             dense_sizes.append(a_tensor.values.shape[a_dense_dims[idx]])
#         else:
#             dense_sizes.append(b_tensor.values.shape[b_dense_dims[idx]])

#     # sparse -> dense conversion
#     exclude = set()
#     for dim in sparse_dims - out_sparse:
#         i = out_sparse_order.index(dim)
#         if dim in out_dense:
#             gather_tensor_name = f"{dim}_g"
#             tensors[gather_tensor_name] = out_indices[:, i]
#             dense_dims.append(f"{gather_tensor_name}[{nnz_index}]")
#             if dim in a_sparse_dims:
#                 dense_sizes.append(a_tensor.shape[a_eqn.index(dim)])
#             else:
#                 dense_sizes.append(b_tensor.shape[b_eqn.index(dim)])

#         exclude.add(i)

#     out_sparse_order = [out_sparse_order[i] for i in range(len(out_sparse_order)) if i not in exclude]
#     out_indices = out_indices[:, [i for i in range(out_indices.shape[1]) if i not in exclude]]

#     # einsum_gs construction
#     tensors["Out_val"] = torch.zeros(dense_sizes)
#     eqn_lhs = f"Out_val[{', '.join([nnz_index] + dense_dims)}]"
#     eqn = eqn_lhs + " += "

#     a_dense_index = [None] * len(a_tensor.values.shape)
#     a_dense_index[0] = a_nnz
#     for dim, i in a_dense_dims.items():
#         if dim in b_sparse_dims:
#             a_dense_index[i] = f"B_crd[{b_nnz}, {b_sparse_dims[dim]}]"
#         else:
#             a_dense_index[i] = dim
#     eqn += f"A_val[{','.join(a_dense_index)}]"
#     eqn += " * "

#     b_dense_index = [None] * len(b_tensor.values.shape)
#     b_dense_index[0] = b_nnz
#     for dim, i in b_dense_dims.items():
#         if dim in a_sparse_dims:
#             b_dense_index[i] = f"A_crd[{a_nnz}, {a_sparse_dims[dim]}]"
#         else:
#             b_dense_index[i] = dim
#     eqn += f"B_val[{','.join(b_dense_index)}]"

#     print(eqn)
#     out_val = einsum_gs(eqn, **tensors)
#     print("intermediate val:")
#     print(out_val)
#     print("intermediate indices:")
#     print(out_indices)

#     # Sparse Coalescing
#     if len(sparse_dims - out_dims) > 0:
#         # perform sparse coalescing if there is a reduction over some sparse dimension.
#         if out_indices.shape[1] > 0:
#             unique_indices, gather_idx = torch.unique(out_indices, dim=0, return_inverse=True)
#         else:
#             unique_indices, gather_idx = torch.zeros((1, 0)), torch.zeros((out_indices.shape[0],), dtype=torch.long)
#         # coalesce_eqn = f"Out[{','.join([f"G[{nnz_index}]"] + dense_dims)}] += {eqn_lhs}"
#         dense_sizes[0] = unique_indices.shape[0]
#         out_val = einsum_gs(coalesce_eqn, Out_val=out_val, G=gather_idx, Out=torch.zeros(dense_sizes))
#         out_indices = unique_indices

#     dimension_format = []
#     dimension_mapping = {}
#     for i, idx in enumerate(out_eqn):
#         if idx in dense_dims:
#             dimension_mapping[i] = dense_dims.index(idx) + 1
#             dimension_format.append(
#                 Dimension(
#                     size=dense_sizes[dense_dims.index(idx) + 1],
#                     format=DimensionFormat.DENSE,
#                 )
#             )
#         else:
#             dimension_mapping[i] = out_sparse_order.index(idx)
#             if idx in a_sparse_dims:
#                 size = a_tensor.shape[a_eqn.index(idx)]
#             else:
#                 size = b_tensor.shape[b_eqn.index(dim)]
#             dimension_format.append(Dimension(size=size, format=DimensionFormat.SPARSE))

#     return SparseTensor(out_indices, out_val, dimension_format, dimension_mapping)


# TODO:
# test with i,i,i->i
#           s,s,d
#           s,d,d

# also test with iij,j->i
#                sds,d


def sparse_einsum(equation: str, out_format: str, *tensors: SparseTensor, table=False) -> SparseTensor:
    """Execute a sparse einsum operation.

    Args:
        equation: Einsum equation in the form "ij,jk->ik"
        out_format: Format of each dimension of the output tensor
        *tensors: Input tensors (mix of sparse and dense)

    Returns:
        Output tensor
    """
    tensor_eqns, out_eqn = parse_einsum_equation(equation)
    # Step 0: Preprocess index sets and sizes
    # ij,j -> i
    # ss,s -> s
    # nm,m -> n

    # tensor_eqns: ["ij", "j"], out_eqn: "j"

    index_sizes = {}
    input_indices = set()
    input_sparse = defaultdict(list)
    # TODO: type hint dict[str (Index), Sequence[SparseTensorDimension]], SparseTensorDimension = Tuple(int (tensor index), int (sparse dimension location))

    # Step 1: Collect sparse dimensions
    # index_sizes => i: n, j: m
    # input_sparse => {i,j}
    # input_dense => {}

    for i, (eqn, tensor) in enumerate(zip(tensor_eqns, tensors)):
        for j, (index, dim) in enumerate(zip(eqn, tensor.dimensions)):
            if index in index_sizes:
                assert index_sizes[index] == dim.size, f"Error: Sizes along index {index} do not match."

            index_sizes[index] = dim.size
            input_indices.add(index)

            if dim.format == DimensionFormat.SPARSE:
                input_sparse[index].append((i, tensor.get_storage_index(j)))

    input_dense = input_indices - set(input_sparse.keys())

    # Step 2: Intersect indicies sparse in input
    # query => T0_0[i0]; T0_1[i0] == T1_0[i1] -> (i0,i1)
    # intersect_data => T0_0 = tensors[0].indices[:, 0], ...
    # int_idx: (nnz, 2[i0,i1])

    intersect_data = {}
    intersect_queries = []
    for index, dim_list in input_sparse.items():
        assert len(dim_list) > 0

        query = []
        for tens, dim in dim_list:
            query.append(f"T{tens}_{dim}[i{tens}]")
            intersect_data[f"T{tens}_{dim}"] = tensors[tens].indices[:, dim]

        # if len(dim_list) == 1:
        #     query *= 2  # hack in unconstrained query

        # TODO: implement chains of equality in tiql, ie A == B == C. for now, we have to do A == B, A == C
        # intersect_queries.append(" == ".join(query))
        intersect_queries.append(f"{query[0]}")
        intersect_queries.extend(f"{query[0]} == {q}" for q in query[1:])
        # intersect_queries.extend(f"{query[0]} == {q}" for q in query)

    if intersect_queries:
        # A[i]; -> torch.arange()

        intersect_query = ", ".join(intersect_queries)
        # print("Intersection:\n", intersect_query)
        # print("with data", intersect_data)
        if table:
            int_idx = table_intersect(intersect_query, **intersect_data)
        else:
            int_idx = intersect(intersect_query, **intersect_data)
        # index_table = tensors[0].indices[:, 0].unsqueeze(0) == tensors[1].indices[:, 0].unsqueeze(1)
        # int_idx = torch.nonzero(index_table)
        # int_idx = torch.zeros((1, 0), dtype=torch.long)
    else:
        int_idx = torch.zeros(1, 0, dtype=torch.long)

    j = 0
    tensor_idx_to_int_idx_mapping = {}  # map each tensor's index in int_idx to a dimension of int_idx
    for i, tensor in enumerate(tensors):
        if not tensor.is_dense:
            tensor_idx_to_int_idx_mapping[i] = j
            j += 1

    # Step 2.5: Construct output indices for sparse -> sparse dimensions
    # out_indices: (nnz, 1)
    out_indices_columns = []
    out_indices_order = []
    for i, ind in enumerate(out_eqn):
        if ind in input_sparse and out_format[i] == "s":
            canon_index_tensor, canon_index_dim = input_sparse[ind][0]
            #  TODO: fix indexing bug
            out_indices_columns.append(
                tensors[canon_index_tensor].indices[
                    int_idx[tensor_idx_to_int_idx_mapping[canon_index_tensor]], canon_index_dim
                ]
            )
            out_indices_order.append(ind)

    if out_indices_columns:
        out_indices = torch.stack(out_indices_columns, dim=1)
    else:
        out_indices = torch.zeros((int_idx.shape[1], 0), dtype=torch.long)
    # Step 3: Perform gather-scatter einsum
    nnz_index = "p"

    # Match einsum data tensors to intersection dimensions, noting
    # only tensors with sparse dimensions appear in the intersection.

    # Out[p] += T0[P0[p]] * T1[P1[p]]
    # einsum_data: T0 = tensors[0].values, T1 = tensors[1].values

    # i,i -> i
    # d,s -> d

    # einsum_data: T0 = tensors[0].values[0], T1 = tensors[1].values
    # Out[p, i_crd[p]] = T0[i_crd[p]] * T1[P1[p]]

    # i,i -> i
    # d,d -> s

    # einsum_data: T0 = tensors[0].values[0], T1 = tensors[1].values[0]
    # Out[p, i] = T0[i] * T1[i]
    # Out[n*p + i] = T0[i] * T1[i] !!!
    # have to make sure out_indices matches

    # i,k -> i,k -> arange() x arange() -> meshgrid()
    # d,d    s,s

    # i(s) -> i(d)
    # Out[G[p], I[P[p]]] = T0[P[p]]

    # crd: [0, 3, 5], val: [1,2,3]
    # G: gather G = [0,0,0]

    einsum_data = {}
    j = 0
    einsum_is_dense = True
    # print("int_idx is", int_idx)
    for i, tensor in enumerate(tensors):
        if not tensor.is_dense:
            einsum_data[f"P{i}"] = int_idx[j]
            j += 1
            einsum_is_dense = False

    # generate unique nnz for output. this both coalesces sparse reduction indices and
    # helps with sparse to dense conversion

    if out_indices.shape[1] > 0:
        out_indices, gather_idx = torch.unique(out_indices, dim=0, return_inverse=True)
    else:
        out_indices, gather_idx = torch.zeros((1, 0), dtype=torch.long), torch.zeros(
            (out_indices.shape[0],), dtype=torch.long
        )

    einsum_data["G"] = gather_idx

    # dimensions that are dense in the input or dense in the output
    einsum_dense_dims = [ind for i, ind in enumerate(out_eqn) if out_format[i] == "d" or ind in input_dense]
    einsum_dense_sizes = [out_indices.shape[0]] + [index_sizes[ind] for ind in einsum_dense_dims]
    einsum_data["Out_val"] = torch.zeros(einsum_dense_sizes)

    einsum_lhs_index = []
    for ind in einsum_dense_dims:
        if ind in input_sparse:
            canon_index_tensor, canon_index_dim = input_sparse[ind][0]
            ind_access = f"T{canon_index_tensor}_{canon_index_dim}"
            einsum_lhs_index.append(f"{ind_access}[P{canon_index_tensor}[{nnz_index}]]")
            einsum_data[ind_access] = tensors[canon_index_tensor].indices[:, canon_index_dim]
        else:
            einsum_lhs_index.append(ind)

    # indirect einsum does not allow an index to only appear on the lhs of equation
    # this occurs naturally with einsums where all inputs are dense,
    # ie. i(d), i(d) -> i(d)
    # produces: Out_val[p, i] = T0[i] * T1[i]
    # The single p on the lhs is forbidden, hence special case logic.
    if einsum_is_dense:
        eqn_lhs = f"Out_val[{', '.join(einsum_lhs_index)}]"
        einsum_data["Out_val"] = einsum_data["Out_val"].squeeze(0)
    else:
        xx = f"G[{nnz_index}]"
        eqn_lhs = f"Out_val[{', '.join([xx] + einsum_lhs_index)}]"

    eqn_rhs = []
    for i, tensor in enumerate(tensors):
        tensor_name = f"T{i}"
        if tensor.is_dense:
            # select 0 on the nnz dimension as it is the only possible value.
            # keep this dimension present as tensor_index is filled, and remove it later
            tensor_index = [None] * len(tensor.values.shape)
            einsum_data[tensor_name] = tensor.values[0]
        else:
            tensor_index = [None] * len(tensor.values.shape)
            tensor_index[0] = f"P{i}[{nnz_index}]"
            einsum_data[tensor_name] = tensor.values

        for j, dim in enumerate(tensor.dimensions):
            if dim.is_sparse:
                continue

            ind = tensor_eqns[i][j]
            if ind in input_sparse:
                canon_index_tensor, canon_index_dim = input_sparse[ind][0]
                ind_access = f"T{canon_index_tensor}_{canon_index_dim}"
                tensor_index[tensor.get_storage_index(j)] = f"{ind_access}[P{canon_index_tensor}[{nnz_index}]]"
                einsum_data[ind_access] = tensors[canon_index_tensor].indices[:, canon_index_dim]
            else:
                tensor_index[tensor.get_storage_index(j)] = ind

        if tensor.is_dense:
            # remove the 0th index dimension as it is fixed at 0
            del tensor_index[0]

        tensor_eqn = f"{tensor_name}[{','.join(tensor_index)}]"
        eqn_rhs.append(tensor_eqn)

    gather_eqn = f"{eqn_lhs} += {' * '.join(eqn_rhs)}"
    # print(gather_eqn)
    # print(einsum_data)
    out_val = einsum_gs(gather_eqn, **einsum_data)
    # print("out was", out_val, "\n\n")

    if einsum_is_dense:
        out_val = out_val.unsqueeze(0)

    # Step 4: Sparsify dense dimensions sparse in input.
    # TODO: use meshgrid
    out_sparse = {ind for i, ind in enumerate(out_eqn) if out_format[i] == "s"}
    i = 0
    while i < len(einsum_dense_dims):
        idx = einsum_dense_dims[i]
        if idx in out_sparse:
            dense_size = index_sizes[idx]
            out_indices_order.append(idx)
            out_indices = torch.cat(
                [
                    out_indices.repeat_interleave(dense_size, dim=0),
                    torch.arange(dense_size).repeat(out_indices.shape[0]).unsqueeze(1),
                ],
                dim=1,
            )
            out_val = out_val.reshape(-1, *[index_sizes[dim] for dim in einsum_dense_dims if dim != idx])

            out_indices_order.append(idx)
            einsum_dense_dims.pop(i)
        else:
            i += 1

    dimension_format = []
    dimension_mapping = {}
    for i, idx in enumerate(out_eqn):
        if out_format[i] == "s":
            dimension_mapping[i] = out_indices_order.index(idx)
            dimension_format.append(Dimension(size=index_sizes[idx], format=DimensionFormat.SPARSE))
        else:
            dimension_mapping[i] = einsum_dense_dims.index(idx)
            dimension_format.append(Dimension(size=index_sizes[idx], format=DimensionFormat.DENSE))

    return SparseTensor(out_indices, out_val, dimension_format, dimension_mapping)
