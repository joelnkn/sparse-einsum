import pytest
import torch
from speinsum.compiler import _coalesce_einsum_indices, _two_operand_einsum, sparse_einsum
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


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "sparse_elementwise_mul",
            "a_eqn": "i",
            "b_eqn": "i",
            "out_eqn": "i",
            "a_dims": [Dimension(8, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(8, DimensionFormat.SPARSE)],
            "out_format": "s",
        },
        {
            "name": "sparse_dot_product",
            "a_eqn": "i",
            "b_eqn": "i",
            "out_eqn": "",
            "a_dims": [
                Dimension(8, DimensionFormat.SPARSE),
            ],
            "b_dims": [
                Dimension(8, DimensionFormat.SPARSE),
            ],
            "out_format": "s",
        },
        # Matrix-vector multiplication
        {
            "name": "sparse_matvec_mul",
            "a_eqn": "ij",
            "b_eqn": "j",
            "out_eqn": "i",
            "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(6, DimensionFormat.SPARSE)],
            "out_format": "s",
        },
        # Matrix-matrix multiplication
        {
            "name": "sparse_matmat_mul",
            "a_eqn": "ik",
            "b_eqn": "kj",
            "out_eqn": "ij",
            "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(6, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.SPARSE)],
            "out_format": "ss",
        },
        # Dense result from sparse matmul
        {
            "name": "sparse_matmat_dense_output",
            "a_eqn": "ik",
            "b_eqn": "kj",
            "out_eqn": "ij",
            "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(6, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(6, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.SPARSE)],
            "out_format": "dd",
        },
        # Broadcasting over dense axis
        {
            "name": "sparse_broadcast_dense_mul",
            "a_eqn": "ij",
            "b_eqn": "j",
            "out_eqn": "ij",
            "a_dims": [Dimension(4, DimensionFormat.SPARSE), Dimension(5, DimensionFormat.DENSE)],
            "b_dims": [Dimension(5, DimensionFormat.DENSE)],
            "out_format": "sd",
        },
        # Mixed dense and sparse outer product
        {
            "name": "sparse_outer_product",
            "a_eqn": "i",
            "b_eqn": "j",
            "out_eqn": "ij",
            "a_dims": [Dimension(4, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(6, DimensionFormat.SPARSE)],
            "out_format": "ss",
        },
        # One input dense, one input sparse
        {
            "name": "dense_sparse_mix",
            "a_eqn": "ij",
            "b_eqn": "ij",
            "out_eqn": "ij",
            "a_dims": [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(3, DimensionFormat.DENSE), Dimension(3, DimensionFormat.SPARSE)],
            "out_format": "ds",
        },
        # No output indices (full reduction)
        {
            "name": "sparse_full_reduction",
            "a_eqn": "ij",
            "b_eqn": "ij",
            "out_eqn": "",
            "a_dims": [Dimension(3, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.SPARSE)],
            "b_dims": [Dimension(3, DimensionFormat.SPARSE), Dimension(3, DimensionFormat.SPARSE)],
            "out_format": "",
        },
        # 3D contraction over middle dimension (sparse-sparse-sparse)
        {
            "name": "sparse_3d_contraction_all_sparse",
            "a_eqn": "ijk",
            "b_eqn": "jkl",
            "out_eqn": "il",
            "a_dims": [
                Dimension(4, DimensionFormat.SPARSE),
                Dimension(5, DimensionFormat.SPARSE),
                Dimension(6, DimensionFormat.SPARSE),
            ],
            "b_dims": [
                Dimension(5, DimensionFormat.SPARSE),
                Dimension(6, DimensionFormat.SPARSE),
                Dimension(7, DimensionFormat.SPARSE),
            ],
            "out_format": "ss",
        },
        # Mixed contraction with dense inner dimension (sparse-dense-sparse)
        {
            "name": "sparse_3d_contraction_inner_dense",
            "a_eqn": "ijk",
            "b_eqn": "jkl",
            "out_eqn": "il",
            "a_dims": [
                Dimension(4, DimensionFormat.SPARSE),
                Dimension(5, DimensionFormat.DENSE),
                Dimension(6, DimensionFormat.SPARSE),
            ],
            "b_dims": [
                Dimension(5, DimensionFormat.DENSE),
                Dimension(6, DimensionFormat.SPARSE),
                Dimension(7, DimensionFormat.SPARSE),
            ],
            "out_format": "ss",
        },
        # Broadcast with one dense axis (outer product style)
        {
            "name": "sparse_dense_broadcast_outer",
            "a_eqn": "ijk",
            "b_eqn": "l",
            "out_eqn": "ijkl",
            "a_dims": [
                Dimension(3, DimensionFormat.SPARSE),
                Dimension(4, DimensionFormat.DENSE),
                Dimension(5, DimensionFormat.SPARSE),
            ],
            "b_dims": [
                Dimension(6, DimensionFormat.SPARSE),
            ],
            "out_format": "sdsd",
        },
        # Diagonal across two sparse dimensions in one input
        {
            "name": "sparse_3d_diagonal_reduce",
            "a_eqn": "iji",
            "b_eqn": "",
            "out_eqn": "j",
            "a_dims": [
                Dimension(4, DimensionFormat.SPARSE),
                Dimension(5, DimensionFormat.DENSE),
                Dimension(4, DimensionFormat.SPARSE),
            ],
            "b_dims": [],
            "out_format": "d",
        },
        # Reduction to scalar (mix of dense and sparse)
        {
            "name": "sparse_highdim_scalar_output",
            "a_eqn": "ijkl",
            "b_eqn": "ijkl",
            "out_eqn": "",
            "a_dims": [
                Dimension(2, DimensionFormat.SPARSE),
                Dimension(3, DimensionFormat.SPARSE),
                Dimension(4, DimensionFormat.DENSE),
                Dimension(5, DimensionFormat.DENSE),
            ],
            "b_dims": [
                Dimension(2, DimensionFormat.SPARSE),
                Dimension(3, DimensionFormat.SPARSE),
                Dimension(4, DimensionFormat.DENSE),
                Dimension(5, DimensionFormat.DENSE),
            ],
            "out_format": "",
        },
    ],
)
def test_two_operand_einsum(test_case):
    print("Performing ", test_case["name"])
    a_tensor = SparseTensor.random_sparse_tensor(test_case["a_dims"], 50)
    b_tensor = SparseTensor.random_sparse_tensor(test_case["b_dims"], 50)

    einsum_eqn = f"{test_case["a_eqn"]}, {test_case["b_eqn"]} -> {test_case["out_eqn"]}"
    expected = torch.einsum(einsum_eqn, a_tensor.to_dense(), b_tensor.to_dense())

    print(a_tensor, b_tensor)
    print(expected)
    out_tensor = _two_operand_einsum(
        test_case["a_eqn"], test_case["b_eqn"], test_case["out_eqn"], a_tensor, b_tensor, test_case["out_format"]
    )

    print(out_tensor)
    assert torch.allclose(out_tensor.to_dense(), expected), "The computed einsum is not as expected"
