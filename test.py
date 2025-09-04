import torch
import torch._inductor.utils
from speinsum.compiler import sparse_einsum
from speinsum.sparse_tensor import SparseTensor
from speinsum.typing import Dimension, DimensionFormat

# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# TORCH_LOGS_FORMAT=“%(levelname)s:%(message)s” TORCH_LOGS="aot_graphs" python test.py
# TORCH_COMPILE_DEBUG=1 python test.py

# Data
if torch.cuda.is_available():
    device = torch.device("cuda")  # Default CUDA device
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

# test_case = {
#     "name": "sparse_elementwise_mul",
#     "equation": "i,i->i",
#     "out_format": "d",
#     "tensor_dims": [[Dimension(10, DimensionFormat.SPARSE)], [Dimension(10, DimensionFormat.SPARSE)]],
# }

test_case = {
    "name": "spmspm",
    "equation": "ik,kj->ij",
    "out_format": "dd",
    "tensor_dims": [
        [Dimension(10**3, DimensionFormat.SPARSE), Dimension(10**3, DimensionFormat.SPARSE)],
        [Dimension(10**3, DimensionFormat.SPARSE), Dimension(10**3, DimensionFormat.SPARSE)],
    ],
}

tensors = [SparseTensor.random_sparse_tensor(dims, 100, device=device) for dims in test_case["tensor_dims"]]
aten = torch.ops.aten

"""
        full_7: "b8[1, 100][100, 1]cpu" = torch.ops.aten.full.default([1, 100], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        bitwise_and_2: "b8[100, 100][100, 1]cpu" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_1, full_7);  bitwise_and_1 = full_7 = None

"""
"""

# TORCH_LOGS_FORMAT="%(levelname)s: %(message)s" TORCH_LOGS="aot_graphs,output_code,post_grad_graphs" python test.py
def extra_check_test(match):
    fill_value = match.kwargs["fill_value"]
    return fill_value


@register_graph_pattern(
    CallFunction(
        aten.bitwise_and.Tensor,
        KeywordArg("t1"),
        CallFunction(
            aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            # _users=MULTIPLE,
        ),
        # _users=MULTIPLE,
    ),
    pass_dict=pass_patterns[1],
    extra_check=extra_check_test,
)
def pointless_bwand_replacement(match: Match, shape, fill_value, device, dtype, t1):
    def repl(x):
        return x

    # only replace the output node, not all nodes
    match.replace_by_example(repl, [t1])
"""

with torch._inductor.utils.fresh_inductor_cache():
    compiled_einsum = torch.compile(sparse_einsum)
    expected = torch.einsum(test_case["equation"], *[t.to_dense() for t in tensors])
    result = compiled_einsum(test_case["equation"], test_case["out_format"], *tensors, table=True)
    # result = sparse_einsum(test_case["equation"], test_case["out_format"], *tensors)
