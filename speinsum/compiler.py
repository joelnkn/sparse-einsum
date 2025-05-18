"""
Core compiler implementation for sparse einsum operations.
"""

from typing import List, Tuple
import torch
from .sparse_tensor import SparseTensor


def parse_einsum_equation(equation: str) -> Tuple[List[str], str]:
    """Parse an einsum equation into input subscripts and output subscript.

    Args:
        equation: String of the form "ij,jk->ik"

    Returns:
        Tuple of (input_subscripts, output_subscript)
        where input_subscripts is a list of strings for each input tensor
        and output_subscript is a string for the output tensor
    """
    equation_sides = equation.split("->")

    inputs = equation_sides[0]
    input_subscripts = [s.strip() for s in inputs.split(",")]

    if len(equation_sides) > 2:
        raise ValueError("Invalid einsum equation.")

    if len(equation_sides) == 1:
        # Implicit output. The output indices are all non-repeated indices in the input, sorted alphabetically.
        included_indices = set()
        excluded_indices = set()

        for index in "".join(input_subscripts):
            if index in included_indices:
                excluded_indices.add(index)
            else:
                included_indices.add(index)

        output = "".join(sorted(included_indices - excluded_indices))
    else:
        # Explicit output.
        output = equation_sides[1]

    if not output:
        raise ValueError("Invalid einsum equation.")

    return input_subscripts, output.strip()


def _coalesce_einsum_indices(
    input_eqn: str, input: SparseTensor
) -> Tuple[str, SparseTensor]:
    """Coalesce repeated indices of a tensor input in some einsum equation
    into an equivalent einsum equation and sparse tensor.

    Args:
        input_eqn: Einsum equation for the input tensor
        input: Input tensor

    Returns:
        Tuple of (einsum equation, coalesced tensor)
    """
    seen_indices = set()
    for index in input_eqn:
        if index in seen_indices:
            # Coalesce repeated indices by taking diagonal.
            ...
        else:
            ...

        seen_indices.add(index)


def _two_operand_einsum(
    a_eqn: str, b_eqn: str, out_eqn: str, a_tensor: SparseTensor, b_tensor: SparseTensor
) -> SparseTensor:
    """Execute a two-operand einsum operation.

    Args:
        a_eqn: Einsum equation for the first tensor
        b_eqn: Einsum equation for the second tensor
    """


def sparse_einsum(
    equation: str, out_format: str, *tensors: SparseTensor
) -> SparseTensor:
    """Execute a sparse einsum operation.

    Args:
        equation: Einsum equation in the form "ij,jk->ik"
        *tensors: Input tensors (mix of sparse and dense)
        optimize: Whether to apply optimizations

    Returns:
        Output tensor
    """
    # TODO: Implement actual compilation logic

    return torch.einsum(equation, *tensors)
