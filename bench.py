import pytest
import torch
import torch._inductor.utils
from speinsum.compiler import sparse_einsum
from speinsum.sparse_tensor import SparseTensor
from speinsum.typing import Dimension, DimensionFormat
import gc
import time
import torch.backends.opt_einsum as opt_einsum

# TODO: benchmark more kernels eg. 2d
# note: use eager mode
# sparsity, output format, ...

# TODO: torch profiler


# torch._dynamo.config.capture_dynamic_output_shape_ops = True


# torch.backends.opt_einsum.enabled = False

# TORCH_LOGS_FORMAT=“%(levelname)s:%(message)s” TORCH_LOGS="graph_breaks" python test.py
# TORCH_COMPILE_DEBUG=1 python test.py

test_case = {
    "name": "sparse_elementwise_mul",
    "equation": "i,i->i",
    "out_format": "d",
    "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.SPARSE)]],
}


def get_tensors(n: int, **kwargs):
    test_case = {
        "name": "sparse_elementwise_mul",
        "equation": "i,i->i",
        "out_format": "d",
        "tensor_dims": [[Dimension(n, DimensionFormat.SPARSE)], [Dimension(n, DimensionFormat.SPARSE)]],
    }
    return [SparseTensor.random_sparse_tensor(dims, n * 0.1) for dims in test_case["tensor_dims"]]


def sparse_einsum_test(compile: bool, table: bool, **kwargs):
    tensors = kwargs["tensors"]
    with torch._inductor.utils.fresh_inductor_cache():
        if compile:
            compiled_einsum = torch.compile(sparse_einsum)
            result = compiled_einsum(test_case["equation"], test_case["out_format"], *tensors, table=table)
        else:
            result = sparse_einsum(test_case["equation"], test_case["out_format"], *tensors, table=table)


tensors = get_tensors(n=int(10e4))


def benchmark(f, arg_list):
    for args in arg_list:
        print(f.__name__, "with params: ", args)
        # %timeit -r3 -n10 f(**args, tensors=tensors)

        f(**args, tensors=tensors)

        start_time = time.time()
        for i in range(3):
            f(**args, tensors=tensors)
        end_time = time.time()
        print(end_time - start_time)
        torch.cuda.empty_cache()
        gc.collect()
        print("...")


print("starting bench \n\n")
benchmark(
    sparse_einsum_test,
    [
        {"n": 10, "compile": False, "table": True},
        {"n": 10, "compile": True, "table": True},
        {"n": 10, "compile": False, "table": False},
        {"n": 10, "compile": True, "table": False},
    ],
)
