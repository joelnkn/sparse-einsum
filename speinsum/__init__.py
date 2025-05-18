"""
SPEinsum: A compiler for sparse einsum operations on PyTorch tensors.
"""

__version__ = "0.1.0"

from .compiler import sparse_einsum
from .sparse_tensor import SparseTensor

__all__ = ["sparse_einsum", "SparseTensor"]
