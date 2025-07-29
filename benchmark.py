import pytest
import torch
import torch._inductor.utils
from speinsum.compiler import sparse_einsum
from speinsum.sparse_tensor import SparseTensor
from speinsum.typing import Dimension, DimensionFormat
import gc
import time
import torch.backends.opt_einsum as opt_einsum
from typing import Callable

# Data
if torch.cuda.is_available():
    device = torch.device("cuda")  # Default CUDA device
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")


def benchmark_func(
    func: Callable, *args, warmup: int = 10, iters: int = 50, device_sync: bool = True
) -> tuple[float, float]:
    """Benchmark a PyTorch function with warmup and return (min_time, avg_time) in seconds."""
    # Warmup
    for _ in range(warmup):
        func(*args)
        if device_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        func(*args)
        if device_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return min(times), sum(times) / len(times)


def profile_func(func: Callable, *args, trace_dir: str = "./log_profile", steps: int = 10) -> None:
    """Profile a PyTorch function and save results to TensorBoard."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(steps):
            func(*args)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"Profile trace saved to {trace_dir}. Run:\n  tensorboard --logdir={trace_dir}")


def benchmark_and_profile(
    func: Callable, *args, warmup: int = 4, iters: int = 20, trace_dir: str = "./log_profile", profile_steps: int = 10
) -> None:
    """Benchmark and profile a PyTorch function."""
    # print(f"\nBenchmarking {func.__name__} ...")
    min_time, avg_time = benchmark_func(func, *args, warmup=warmup, iters=iters)
    print(f"Min time: {min_time*1e3:.3f} ms | Avg time: {avg_time*1e3:.3f} ms")

    # print(f"\nProfiling {func.__name__} (saving trace to {trace_dir}) ...")
    # profile_func(func, *args, trace_dir=trace_dir, steps=profile_steps)


def run_sparse_einsum(test_case, tensors, table):
    return sparse_einsum(test_case["equation"], test_case["out_format"], *tensors, table=table)


# test_case = {
#     "name": "sparse_elementwise_mul",
#     "equation": "i,i->i",
#     "out_format": "d",
#     "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.SPARSE)]],
# }


test_cases = [
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
]

for test_case in test_cases:
    print(f"\nBenchmarking {test_case['name']}")
    n = 10e5
    tensors = [SparseTensor.random_sparse_tensor(dims, n * 0.1) for dims in test_case["tensor_dims"]]

    # Eager mode function
    eager_func = run_sparse_einsum

    # Compiled function (with torch.compile)
    compiled_func = torch.compile(run_sparse_einsum)

    # Run benchmarks
    table = True
    print("Table Eager")
    benchmark_and_profile(eager_func, test_case, tensors, table, trace_dir="./.log_eager")
    print("Table Compiled")
    benchmark_and_profile(compiled_func, test_case, tensors, table, trace_dir="./.log_compiled")

    table = False
    print("Binary Search Eager")
    benchmark_and_profile(eager_func, test_case, tensors, table, trace_dir="./.log_eager")
    print("Binary Search Compiled")
    benchmark_and_profile(compiled_func, test_case, tensors, table, trace_dir="./.log_compiled")

    # print(f"Eager mode:    min = {eager_min*1e3:.3f} ms, avg = {eager_avg*1e3:.3f} ms")
    # print(f"torch.compile: min = {compiled_min*1e3:.3f} ms, avg = {compiled_avg*1e3:.3f} ms")
