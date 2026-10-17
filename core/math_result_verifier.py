"""Offline-only oracle contract for one explicit math_calculate task.

No production callsite is installed. The caller must bind each returned result
one-to-one to the exact request before invoking this callback. ToolExecutionResult
has no request identifier, so this module alone cannot establish that binding.
The evaluator marker proves callback invocation, not external authentication.
The callback computes a score only: it neither records strategy credit nor updates
strategy state. Any monitor/adaptation effects depend on the caller, not this module.
"""

import re
from decimal import Context, Decimal, Inexact, Rounded, localcontext
from typing import Callable, Optional


# ASCII, canonical, bounded, nonnegative integers; no whitespace normalization,
# parser reuse, expressions in prose, floats, function calls, or arbitrary code.
_ADDITION = re.compile(r"(0|[1-9][0-9]{0,11}) \+ (0|[1-9][0-9]{0,11})", re.ASCII)
ORACLE_VERSION = "decimal-integer-addition-v1"
TASK_TYPE = "tool_integer_addition"


def make_result_evaluator(expression: str) -> Optional[Callable[[object], float]]:
    """Return a pure request-bound 0/1 evaluator, or None outside the oracle domain.

    Ground truth is decimal addition of the request operands, never a value or
    success assertion supplied by the system under evaluation. An isolated
    13-digit context suffices for the sum of two at-most-12-digit integers.
    All inexact/rounded arithmetic fails closed. No tolerance is tuned.
    Unsupported requests MUST NOT be passed to a heuristic substitute.
    Calling the returned function has no persistence, production, or adaptation
    side effects; callers must keep it behind a sealed, offline evaluation phase.
    """
    if type(expression) is not str or len(expression) > 27:
        return None
    match = _ADDITION.fullmatch(expression)
    if match is None:
        return None
    with localcontext(Context(prec=13)) as context:
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        expected = context.add(Decimal(match[1]), Decimal(match[2]))

    def result_evaluator(result: object) -> float:
        # Lazy import avoids loading the solver to compute the oracle.
        from core.constitutional_brain import ToolExecutionResult

        if type(result) is not ToolExecutionResult:
            raise ValueError("an exact ToolExecutionResult is required")
        if type(result.tool_name) is not str or result.tool_name != "math_calculate":
            raise ValueError("result is not from the bound math_calculate tool")
        if type(result.success) is not bool:
            raise ValueError("success must be a built-in bool")
        if result.success is False:
            return 0.0
        # Reject bool, floats and arbitrary equality/conversion hooks, even when
        # numerically equal. The operational integer-addition executor returns int.
        if type(result.result) is not int or result.error is not None:
            return 0.0
        if result.result < 0 or result.result.bit_length() > 41:
            return 0.0
        return float(Decimal(result.result) == expected)

    return result_evaluator
