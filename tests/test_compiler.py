import pytest
import torch
from speinsum.compiler import _coalesce_einsum_indices, sparse_einsum, parse_einsum_equation
from speinsum.sparse_tensor import SparseTensor
from speinsum.typing import Dimension, DimensionFormat


def test_parse_einsum_equation():
    from speinsum.compiler import parse_einsum_equation

    # Test basic matrix multiplication case
    inputs, output = parse_einsum_equation("ij,jk->ik")
    assert inputs == ["ij", "jk"]
    assert output == "ik"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "simple_sparse_repeated",
            "indices": [[0, 0], [1, 1], [2, 2]],
            "values": [1.0, 2.0, 3.0],
            "dimensions": [
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.SPARSE),
            ],
            "input_eqn": "ii",
        },
        {
            "name": "mixed_sparse_dense",
            "indices": [[0, 0], [1, 1], [2, 2]],
            "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "dimensions": [
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.SPARSE),
                (2, DimensionFormat.DENSE),
            ],
            "input_eqn": "iij",
        },
        {
            "name": "mixed_sparse_dense_diagonal",
            "indices": [[0, 0], [1, 1], [2, 2]],
            "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "dimensions": [
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.DENSE),
            ],
            "input_eqn": "iii",
        },
        {
            "name": "no_repeated_indices",
            "indices": [[0, 0], [1, 1], [2, 2]],
            "values": [1.0, 2.0, 3.0],
            "dimensions": [
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.SPARSE),
            ],
            "input_eqn": "ij",
        },
        {
            "name": "invalid_indices",
            "indices": [[0, 1], [1, 2], [2, 0]],
            "values": [1.0, 2.0, 3.0],
            "dimensions": [
                (3, DimensionFormat.SPARSE),
                (3, DimensionFormat.SPARSE),
            ],
            "input_eqn": "ii",
        },
        {
            "name": "dense_diagonal",
            "indices": [[0]],
            "values": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            "dimensions": [
                (1, DimensionFormat.SPARSE),
                (3, DimensionFormat.DENSE),
                (3, DimensionFormat.DENSE),
            ],
            "input_eqn": "jii",
        },
    ],
)
def test_coalesce_einsum_indices(test_case):
    """Test coalescing of einsum indices with various tensor configurations."""
    # Create tensor
    tensor = SparseTensor(
        indices=torch.tensor(test_case["indices"]),
        values=torch.tensor(test_case["values"]),
        dimensions=tuple(Dimension(size, fmt) for size, fmt in test_case["dimensions"]),
    )

    # Run coalescing
    new_eqn, new_tensor = _coalesce_einsum_indices(test_case["input_eqn"], tensor)

    # Verify results
    assert len(new_eqn) == len(set(new_eqn))
    assert set(new_eqn) == set(test_case["input_eqn"])

    # Verify values are preserved
    # TODO: add test case logic here. probably check that a random value on the diagonal is preserved.

    print("\n", test_case["input_eqn"], new_eqn)
    print(tensor)
    print(new_tensor)


# @pytest.mark.parametrize(
#     "test_case",
#     [
#         {
#             "name": "sparse_elementwise_mul",
#             "a_eqn": "i",
#             "b_eqn": "i",
#             "out_eqn": "i",
#             "a_dims": [Dimension(8, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(8, DimensionFormat.SPARSE)],
#             "out_format": "s",
#         },
#         {
#             "name": "sparse_dot_product",
#             "a_eqn": "i",
#             "b_eqn": "i",
#             "out_eqn": "",
#             "a_dims": [
#                 Dimension(8, DimensionFormat.SPARSE),
#             ],
#             "b_dims": [
#                 Dimension(8, DimensionFormat.SPARSE),
#             ],
#             "out_format": "s",
#         },
#         # Matrix-vector multiplication
#         {
#             "name": "sparse_matvec_mul",
#             "a_eqn": "ij",
#             "b_eqn": "j",
#             "out_eqn": "i",
#             "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(6, DimensionFormat.SPARSE)],
#             "out_format": "s",
#         },
#         # Matrix-matrix multiplication
#         {
#             "name": "sparse_matmat_mul",
#             "a_eqn": "ik",
#             "b_eqn": "kj",
#             "out_eqn": "ij",
#             "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(6, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.SPARSE)],
#             "out_format": "ss",
#         },
#         # Dense result from sparse matmul
#         {
#             "name": "sparse_matmat_dense_output",
#             "a_eqn": "ik",
#             "b_eqn": "kj",
#             "out_eqn": "ij",
#             "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(6, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.SPARSE)],
#             "out_format": "dd",
#         },
#         # Broadcasting over dense axis
#         {
#             "name": "sparse_broadcast_dense_mul",
#             "a_eqn": "ij",
#             "b_eqn": "j",
#             "out_eqn": "ij",
#             "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.DENSE)],
#             "b_dims": [Dimension(5, DimensionFormat.DENSE)],
#             "out_format": "sd",
#         },
#         # Mixed dense and sparse outer product
#         {
#             "name": "sparse_outer_product",
#             "a_eqn": "i",
#             "b_eqn": "j",
#             "out_eqn": "ij",
#             "a_dims": [Dimension(4, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(6, DimensionFormat.SPARSE)],
#             "out_format": "ss",
#         },
#         # One input dense, one input sparse
#         {
#             "name": "dense_sparse_mix",
#             "a_eqn": "ij",
#             "b_eqn": "ij",
#             "out_eqn": "ij",
#             "a_dims": [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.SPARSE)],
#             "out_format": "ds",
#         },
#         # No output indices (full reduction)
#         {
#             "name": "sparse_full_reduction",
#             "a_eqn": "ij",
#             "b_eqn": "ij",
#             "out_eqn": "",
#             "a_dims": [Dimension(3, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.SPARSE)],
#             "b_dims": [Dimension(3, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.SPARSE)],
#             "out_format": "",
#         },
#         # 3D contraction over middle dimension (sparse-sparse-sparse)
#         {
#             "name": "sparse_3d_contraction_all_sparse",
#             "a_eqn": "ijk",
#             "b_eqn": "jkl",
#             "out_eqn": "il",
#             "a_dims": [
#                 Dimension(4, DimensionFormat.SPARSE),
#                 Dimension(5, DimensionFormat.SPARSE),
#                 Dimension(6, DimensionFormat.SPARSE),
#             ],
#             "b_dims": [
#                 Dimension(5, DimensionFormat.SPARSE),
#                 Dimension(6, DimensionFormat.SPARSE),
#                 Dimension(7, DimensionFormat.SPARSE),
#             ],
#             "out_format": "ss",
#         },
#         # Mixed contraction with dense inner dimension (sparse-dense-sparse)
#         {
#             "name": "sparse_3d_contraction_inner_dense",
#             "a_eqn": "ijk",
#             "b_eqn": "jkl",
#             "out_eqn": "il",
#             "a_dims": [
#                 Dimension(4, DimensionFormat.SPARSE),
#                 Dimension(5, DimensionFormat.DENSE),
#                 Dimension(6, DimensionFormat.SPARSE),
#             ],
#             "b_dims": [
#                 Dimension(5, DimensionFormat.DENSE),
#                 Dimension(6, DimensionFormat.SPARSE),
#                 Dimension(7, DimensionFormat.SPARSE),
#             ],
#             "out_format": "ss",
#         },
#         # Broadcast with one dense axis (outer product style)
#         {
#             "name": "sparse_dense_broadcast_outer",
#             "a_eqn": "ijk",
#             "b_eqn": "l",
#             "out_eqn": "ijkl",
#             "a_dims": [
#                 Dimension(3, DimensionFormat.SPARSE),
#                 Dimension(4, DimensionFormat.DENSE),
#                 Dimension(5, DimensionFormat.SPARSE),
#             ],
#             "b_dims": [
#                 Dimension(6, DimensionFormat.SPARSE),
#             ],
#             "out_format": "sdsd",
#         },
#         # Diagonal across two sparse dimensions in one input
#         {
#             "name": "sparse_3d_diagonal_reduce",
#             "a_eqn": "iji",
#             "b_eqn": "",
#             "out_eqn": "j",
#             "a_dims": [
#                 Dimension(4, DimensionFormat.SPARSE),
#                 Dimension(5, DimensionFormat.DENSE),
#                 Dimension(4, DimensionFormat.SPARSE),
#             ],
#             "b_dims": [],
#             "out_format": "d",
#         },
#         # Reduction to scalar (mix of dense and sparse)
#         {
#             "name": "sparse_highdim_scalar_output",
#             "a_eqn": "ijkl",
#             "b_eqn": "ijkl",
#             "out_eqn": "",
#             "a_dims": [
#                 Dimension(2, DimensionFormat.SPARSE),
#                 Dimension(3, DimensionFormat.SPARSE),
#                 Dimension(4, DimensionFormat.DENSE),
#                 Dimension(5, DimensionFormat.DENSE),
#             ],
#             "b_dims": [
#                 Dimension(2, DimensionFormat.SPARSE),
#                 Dimension(3, DimensionFormat.SPARSE),
#                 Dimension(4, DimensionFormat.DENSE),
#                 Dimension(5, DimensionFormat.DENSE),
#             ],
#             "out_format": "",
#         },
# ],
# )
# def test_two_operand_einsum(test_case):
#     print("Performing ", test_case["name"])
#     a_tensor = SparseTensor.random_sparse_tensor(test_case["a_dims"], 50)
#     b_tensor = SparseTensor.random_sparse_tensor(test_case["b_dims"], 50)

#     einsum_eqn = f"{test_case["a_eqn"]}, {test_case["b_eqn"]} -> {test_case["out_eqn"]}"
#     expected = torch.einsum(einsum_eqn, a_tensor.to_dense(), b_tensor.to_dense())

#     print(a_tensor, b_tensor)
#     print(expected)
#     out_tensor = _two_operand_einsum(
#         test_case["a_eqn"], test_case["b_eqn"], test_case["out_eqn"], a_tensor, b_tensor, test_case["out_format"]
#     )

#     print(out_tensor)
#     assert torch.allclose(out_tensor.to_dense(), expected), "The computed einsum is not as expected"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "sparse_elementwise_mul",
            "equation": "i,i->i",
            "out_format": "d",
            "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.SPARSE)]],
        },
        {
            "name": "dense_elementwise_mul",
            "equation": "i,i->i",
            "out_format": "d",
            "tensor_dims": [[Dimension(10, DimensionFormat.DENSE)], [Dimension(10, DimensionFormat.DENSE)]],
        },
        {
            "name": "sparse_dense_elementwise_mul",
            "equation": "i,i->i",
            "out_format": "s",
            "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.DENSE)]],
        },
        {
            "name": "sparse_dense_broadcast_mul",
            "equation": "ij,j->ij",
            "out_format": "sd",
            "tensor_dims": [
                [Dimension(8, DimensionFormat.SPARSE), Dimension(4, DimensionFormat.DENSE)],
                [Dimension(4, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "sparse_sparse_reduce_sum",
            "equation": "ij->i",
            "out_format": "s",
            "tensor_dims": [[Dimension(5, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)]],
        },
        {
            "name": "sparse_dense_matvec",
            "equation": "ij,j->i",
            "out_format": "s",
            "tensor_dims": [
                [Dimension(7, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.DENSE)],
                [Dimension(3, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "sparse_dense_matmul",
            "equation": "ik,kj->ij",
            "out_format": "sd",
            "tensor_dims": [
                [Dimension(6, DimensionFormat.SPARSE), Dimension(4, DimensionFormat.DENSE)],
                [Dimension(4, DimensionFormat.DENSE), Dimension(5, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "sparse_sparse_contract",
            "equation": "ik,kj->ij",
            "out_format": "ss",
            "tensor_dims": [
                [Dimension(6, DimensionFormat.SPARSE), Dimension(4, DimensionFormat.SPARSE)],
                [Dimension(4, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.SPARSE)],
            ],
        },
        {
            "name": "sparse_batch_matmul",
            "equation": "bij,bjk->bik",
            "out_format": "ssd",
            "tensor_dims": [
                [
                    Dimension(2, DimensionFormat.SPARSE),
                    Dimension(3, DimensionFormat.SPARSE),
                    Dimension(4, DimensionFormat.DENSE),
                ],
                [
                    Dimension(2, DimensionFormat.SPARSE),
                    Dimension(4, DimensionFormat.DENSE),
                    Dimension(5, DimensionFormat.DENSE),
                ],
            ],
        },
        {
            "name": "broadcast_dense_dense_outer",
            "equation": "i,j->ij",
            "out_format": "dd",
            "tensor_dims": [
                [Dimension(3, DimensionFormat.DENSE)],
                [Dimension(4, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "sparse_outer_product",
            "equation": "i,j->ij",
            "out_format": "ss",
            "tensor_dims": [
                [Dimension(3, DimensionFormat.SPARSE)],
                [Dimension(4, DimensionFormat.SPARSE)],
            ],
        },
        {
            "name": "sparse_dense_outer_product",
            "equation": "i,j->ij",
            "out_format": "sd",
            "tensor_dims": [
                [Dimension(3, DimensionFormat.SPARSE)],
                [Dimension(4, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "triple_elementwise_ssd",
            "equation": "i,i,i->i",
            "out_format": "s",
            "tensor_dims": [
                [Dimension(10, DimensionFormat.SPARSE)],
                [Dimension(10, DimensionFormat.SPARSE)],
                [Dimension(10, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "triple_elementwise_sdd",
            "equation": "i,i,i->i",
            "out_format": "s",
            "tensor_dims": [
                [Dimension(10, DimensionFormat.SPARSE)],
                [Dimension(10, DimensionFormat.DENSE)],
                [Dimension(10, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "triple_elementwise_ddd",
            "equation": "i,i,i->i",
            "out_format": "d",
            "tensor_dims": [
                [Dimension(10, DimensionFormat.DENSE)],
                [Dimension(10, DimensionFormat.DENSE)],
                [Dimension(10, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "reduction_mixed_sds_d",
            "equation": "iij,j->i",
            "out_format": "s",
            "tensor_dims": [
                [
                    Dimension(4, DimensionFormat.SPARSE),
                    Dimension(4, DimensionFormat.DENSE),
                    Dimension(5, DimensionFormat.SPARSE),
                ],
                [Dimension(5, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "reduction_mixed_dsd_d",
            "equation": "iij,j->i",
            "out_format": "d",
            "tensor_dims": [
                [
                    Dimension(4, DimensionFormat.DENSE),
                    Dimension(4, DimensionFormat.SPARSE),
                    Dimension(5, DimensionFormat.DENSE),
                ],
                [Dimension(5, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "triple_outer_product_ssd",
            "equation": "i,j,k->ijk",
            "out_format": "ssd",
            "tensor_dims": [
                [Dimension(3, DimensionFormat.SPARSE)],
                [Dimension(4, DimensionFormat.SPARSE)],
                [Dimension(2, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "triangular_contraction",
            "equation": "ij,jk,ki->",
            "out_format": "",
            "tensor_dims": [
                [Dimension(3, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.DENSE)],
                [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.DENSE)],
                [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.DENSE)],
            ],
        },
        {
            "name": "broadcasted_batch_matmul",
            "equation": "bij,bjk,bki->b",
            "out_format": "s",
            "tensor_dims": [
                [
                    Dimension(2, DimensionFormat.SPARSE),
                    Dimension(3, DimensionFormat.DENSE),
                    Dimension(4, DimensionFormat.DENSE),
                ],
                [
                    Dimension(2, DimensionFormat.SPARSE),
                    Dimension(4, DimensionFormat.DENSE),
                    Dimension(5, DimensionFormat.DENSE),
                ],
                [
                    Dimension(2, DimensionFormat.SPARSE),
                    Dimension(5, DimensionFormat.DENSE),
                    Dimension(3, DimensionFormat.DENSE),
                ],
            ],
        },
    ],
)
def test_sparse_einsum(test_case):
    print(f"Performing {test_case["name"]}")
    tensors = [
        SparseTensor.random_sparse_tensor(dims, 50, device=torch.device("cpu")) for dims in test_case["tensor_dims"]
    ]

    # compiled_einsum = torch.compile(sparse_einsum)
    result = sparse_einsum(test_case["equation"], test_case["out_format"], *tensors)
    # result = compiled_einsum(test_case["equation"], test_case["out_format"], *tensors)
    expected = torch.einsum(test_case["equation"], *[t.to_dense() for t in tensors])

    assert torch.allclose(
        result.to_dense(), expected, atol=1e-4
    ), f"The computed einsum is not as expected.\nExpected: {expected} \n\nGot {result}"


# TODO : test on all continuous tensor test cases


@pytest.mark.parametrize("out_format", ["s", "d"])
@pytest.mark.parametrize(
    "equation, tensors, formats",
    [
        ("ik,kj->ij", [torch.randn(2, 4), torch.randn(4, 3)], ["ss", "ds"]),
        ("ijkl->ilkj", [torch.randn(6, 3, 8, 6)], ["ssss"]),
        ("i,i->i", [torch.randn(32), torch.randn(32)], ["s", "s"]),
        # ("ijkl->ilkj", [torch.randn(16, 32, 48, 64)], ["ssss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("id,dj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("xy,yz->xz", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("ef,fj->ej", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["sd", "ss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ss"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(64, 64), torch.randn(64, 64)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(128, 64), torch.randn(64, 32)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ik,kj->ij", [torch.randn(32, 16), torch.randn(16, 8)], ["ds", "ds"]),
        ("ij,ik,kj->ij", [torch.randn(32, 8), torch.randn(32, 16), torch.randn(16, 8)], ["ss", "dd", "dd"]),
        ("ij,ik,kj->ij", [torch.randn(32, 8), torch.randn(32, 16), torch.randn(16, 8)], ["ss", "sd", "dd"]),
        ("ij,ik,kj->ij", [torch.randn(32, 8), torch.randn(32, 16), torch.randn(16, 8)], ["ss", "ds", "dd"]),
        # # ("ij,ik,kj->ij", [torch.randn(256, 256), torch.randn(256, 256), torch.randn(256, 256)], ["sd", "ds", "ds"]),
        ("i,i,i->i", [torch.randn(5), torch.randn(5), torch.randn(5)], ["s", "s", "d"]),
        ("i,i,i->i", [torch.randn(5), torch.randn(5), torch.randn(5)], ["d", "s", "d"]),  # TODO: this test fails
        ("i,i,i->i", [torch.randn(5), torch.randn(5), torch.randn(5)], ["d", "s", "s"]),
        ("i,i,i->i", [torch.randn(5), torch.randn(5), torch.randn(5)], ["s", "s", "s"]),
        # ("i,i,i,i->i", [torch.randn(100), torch.randn(100), torch.randn(100), torch.randn(100)], ["s", "s", "s", "s"]),
        ("ij,ij->ij", [torch.randn(10, 10), torch.randn(10, 10)], ["ss", "ss"]),
        ("ij,ij,ij->ij", [torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10)], ["ss", "ss", "ss"]),
        ("ij,ij,ij->ij", [torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10)], ["ss", "ss", "sd"]),
        ("i,i,j,j->ij", [torch.randn(10), torch.randn(10), torch.randn(10), torch.randn(10)], ["s", "s", "s", "s"]),
        (
            "ij,i,j,j->ij",
            [torch.randn(10, 10), torch.randn(10), torch.randn(10), torch.randn(10)],
            ["ss", "s", "s", "s"],
        ),
        ("i,i,j,j->ij", [torch.randn(10), torch.randn(10), torch.randn(10), torch.randn(10)], ["s", "s", "d", "s"]),
        ("i,i,j,j->ij", [torch.randn(10), torch.randn(10), torch.randn(10), torch.randn(10)], ["s", "s", "d", "d"]),
        ("ikl,lj,kj->ij", [torch.randn(32, 32, 32), torch.randn(32, 32), torch.randn(32, 32)], ["sss", "dd", "dd"]),
        ("ikl,ij,lj->kj", [torch.randn(32, 32, 32), torch.randn(32, 32), torch.randn(32, 32)], ["sss", "dd", "dd"]),
        ("ikl,ij,kj->lj", [torch.randn(32, 32, 32), torch.randn(32, 32), torch.randn(32, 32)], ["sss", "dd", "dd"]),
        ("ikl,lj,kj->ij", [torch.randn(32, 32, 32), torch.randn(32, 32), torch.randn(32, 32)], ["sss", "sd", "dd"]),
        ("ikl,lj,kj->ij", [torch.randn(32, 32, 32), torch.randn(32, 32), torch.randn(32, 32)], ["sss", "sd", "sd"]),
        ("ikl,lj,kj->ij", [torch.randn(16, 16, 16), torch.randn(16, 16), torch.randn(16, 16)], ["sss", "ss", "ss"]),
        ("ii->i", [torch.randn(32, 32)], ["ss"]),
        ("iii->i", [torch.randn(32, 32, 32)], ["sss"]),
        ("iij->ij", [torch.randn(32, 32, 32)], ["ssd"]),
        ("ii,jj->ij", [torch.randn(32, 32), torch.randn(32, 32)], ["ss", "ss"]),
        ("ij,ijk->i", [torch.randn(16, 16), torch.randn(16, 16, 16)], ["sd", "dss"]),
        ("ij,ijk->i", [torch.randn(16, 16), torch.randn(16, 16, 16)], ["ss", "sss"]),
        ("ij,ijk->i", [torch.randn(16, 16), torch.randn(16, 16, 16)], ["sd", "sdd"]),
        ("ik,k->i", [torch.randn(16, 16), torch.randn(16)], ["ss", "d"]),
        ("ik,k->i", [torch.randn(16, 16), torch.randn(16)], ["sd", "d"]),
        ("ik,k->i", [torch.randn(16, 16), torch.randn(16)], ["ds", "s"]),
        ("ik,k->i", [torch.randn(16, 16), torch.randn(16)], ["ss", "s"]),
        ("i->", [torch.randn(32)], ["s"]),
        ("ik,k->", [torch.randn(16, 32), torch.randn(32)], ["ss", "s"]),
        ("ik,k->", [torch.randn(16, 32), torch.randn(32)], ["sd", "s"]),
        ("ik,kj->", [torch.randn(16, 32), torch.randn(32, 24)], ["ss", "ss"]),
        ("ij,ik,kj->", [torch.randn(16, 24), torch.randn(16, 32), torch.randn(32, 24)], ["ss", "ss", "ss"]),
        ("ii->", [torch.randn(16, 16)], ["ss"]),
        ("ij->", [torch.randn(16, 16)], ["ss"]),
    ],
)
def test_continuous_cases(equation, tensors, formats, out_format):
    # Construct sparse tensors from the given dense ones and format info
    def convert_format(fmt):
        return [DimensionFormat.SPARSE if f == "s" else DimensionFormat.DENSE for f in fmt]

    sparse_tensors = [SparseTensor.from_dense(t, convert_format(fmt)) for t, fmt in zip(tensors, formats)]
    # print("\n\n\nGOON")
    # print(tensors)
    # print("\nTHEN\n", sparse_tensors)

    # Extract output format from the formats of the inputs if needed
    # Or infer based on `equation`, for now assume same format as first tensor
    out_format = out_format * len(parse_einsum_equation(equation)[1])

    result = sparse_einsum(equation, out_format, *sparse_tensors)
    expected = torch.einsum(equation, *tensors)

    assert torch.allclose(
        result.to_dense(), expected, atol=1e-4
    ), f"Einsum failed for {equation}\nExpected:\n{expected}\n\nGot:\n{result.to_dense()}"
