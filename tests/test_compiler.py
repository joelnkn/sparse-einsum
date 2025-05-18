import pytest
import torch
from speinsum import compile_sparse_einsum


def test_parse_einsum_equation():
    from speinsum.compiler import parse_einsum_equation

    # Test basic matrix multiplication case
    inputs, output_list, output = parse_einsum_equation("ij,jk->ik")
    assert inputs == ["ij", "jk"]
    assert output_list == ["i", "k"]
    assert output == "ik"


def test_basic_dense_einsum():
    # Test with dense tensors first
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)

    # Compare with PyTorch's einsum
    expected = torch.einsum("ij,jk->ik", a, b)
    result = compile_sparse_einsum("ij,jk->ik", a, b)

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
    result = compile_sparse_einsum("ij,jk->ik", sparse, dense)

    assert torch.allclose(result, expected)
