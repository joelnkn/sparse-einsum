import pytest
import torch
from speinsum.compiler import _coalesce_einsum_indices, sparse_einsum
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


def test_basic_dense_einsum():
    # Test with dense tensors first
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)

    # Compare with PyTorch's einsum
    expected = torch.einsum("ij,jk->ik", a, b)
    result = sparse_einsum("ij,jk->ik", "ik", a, b)

    assert torch.allclose(result, expected)


def test_sparse_dense_einsum():
    # Create a sparse matrix
    indices = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 2D indices
    values = torch.tensor([1.0, 2.0, 3.0])
    sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

    # Dense matrix
    dense = torch.randn(3, 4)

    # Compare with PyTorch's einsum (after converting sparse to dense)
    expected = torch.einsum("ij,jk->ik", sparse.to_dense(), dense)
    result = sparse_einsum("ij,jk->ik", "ik", sparse, dense)

    assert torch.allclose(result, expected)
