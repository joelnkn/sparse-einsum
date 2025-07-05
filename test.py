import pytest
import torch
from speinsum.compiler import _coalesce_einsum_indices, _two_operand_einsum, sparse_einsum, parse_einsum_equation
from speinsum.sparse_tensor import SparseTensor
from speinsum.typing import Dimension, DimensionFormat


# TORCH_LOGS_FORMAT=“%(levelname)s:%(message)s” TORCH_LOGS="graph_breaks" python test.py

test_case = {
    "name": "sparse_elementwise_mul",
    "equation": "i,i->i",
    "out_format": "d",
    "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.SPARSE)]],
}
tensors = [SparseTensor.random_sparse_tensor(dims, 50) for dims in test_case["tensor_dims"]]

compiled_einsum = torch.compile(sparse_einsum)
# result = sparse_einsum(test_case["equation"], test_case["out_format"], *tensors)
result = compiled_einsum(test_case["equation"], test_case["out_format"], *tensors)
expected = torch.einsum(test_case["equation"], *[t.to_dense() for t in tensors])
