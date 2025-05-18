# SPEinsum: Sparse Einsum Compiler for PyTorch

A compiler for optimizing sparse Einstein summation operations on PyTorch tensors. This library integrates with the PyTorch compiler infrastructure to provide efficient compilation of sparse einsum operations.

## Features

- Sparse einsum operation compilation
- Integration with PyTorch's compiler infrastructure
- Optimized tensor operations for sparse data

## Installation

```bash
pip install .
```

For development installation:

```bash
pip install -e ".[dev]"
```

## Usage

Basic example:

```python
import torch
from speinsum import compile_sparse_einsum

# Example sparse tensor
sparse_tensor = torch.sparse_coo_tensor(...)
dense_tensor = torch.randn(...)

# Compile and run sparse einsum
result = compile_sparse_einsum("ij,jk->ik", sparse_tensor, dense_tensor)
```

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`

## License

MIT License
