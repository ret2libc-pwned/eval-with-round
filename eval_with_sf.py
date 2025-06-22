# eval_with_sf.py
"""AP-Chem-style significant-figure evaluator

This module exposes a single helper — ``eval_with_sf`` — that takes an algebraic
expression written with the standard +, −, ×, and ÷ symbols (use * and / in the
string) and returns the correctly rounded result as a string.

Rules implemented
-----------------
* **Multiplication & division** – the answer keeps the same *significant-figure* 
  count as the operand with the fewest sig-figs.
* **Addition & subtraction** – the answer is rounded to the smallest *number of
  decimal places* found in the operands.
* Mixed, parenthesised expressions are evaluated left-to-right using normal
  mathematical precedence, applying the correct rounding rule *after every
  individual operation* (so intermediate values never drag along extra, bogus
  precision).

Example
-------
>>> eval_with_sf("(12.01 + 1.2) / 3.44")
'3.86'  # (12.0 + 1.2 → 13.2 → one decimal) ÷ 3.44 (*3 s.f.*) → 3.837… → 3.86
"""
from __future__ import annotations

import ast
import operator as _op
import re
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Callable, Union

# Give ourselves plenty of head-room so that quantise operations never run out
# of working precision before we round deliberately.
getcontext().prec = 50

# ──────────────────────────────────────────────────────────────────────────────
# Helpers to interrogate and round a numeric token
# ──────────────────────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=$|[^(eE)\d.])")

class SFNumber:
    """A value tagged with its sig-fig and decimal-place metadata."""

    __slots__ = ("value", "sig_figs", "dec_places")

    value: Decimal               # Numeric value (full precision)
    sig_figs: int                # How many significant figures this number has
    dec_places: int              # Digits *after* the decimal point (≥0)

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, value: Decimal, sig_figs: int, dec_places: int):
        self.value = value
        self.sig_figs = sig_figs
        self.dec_places = dec_places

    @classmethod
    def from_literal(cls, literal: str) -> "SFNumber":
        """Build an :class:`SFNumber` directly from a token in the user string."""
        val = Decimal(literal)
        return cls(val, _count_sig_figs(literal), _count_dec_places(literal))

    # ------------------------------------------------------------------
    # Operators applying sig-fig/decimal-place rules *immediately*
    # ------------------------------------------------------------------
    def _round_to_sf(self, sf: int) -> Decimal:
        if self.value.is_zero():  # 0 has *infinite* trailing zero sig-figs
            return Decimal(0)
        magnitude = self.value.adjusted()       # position of first digit
        quant_exp = -(sf - 1 - magnitude)       # exponent for Decimal.quantize
        quantiser = Decimal(f"1e{quant_exp}")
        return self.value.quantize(quantiser, ROUND_HALF_UP)

    def _round_to_dp(self, dp: int) -> Decimal:
        quantiser = Decimal(f"1e-{dp}")
        return self.value.quantize(quantiser, ROUND_HALF_UP)

    # Addition / Subtraction → decimal-places rule ----------------------
    def _addsub(self, other: "SFNumber", op: Callable[[Decimal, Decimal], Decimal]) -> "SFNumber":
        raw = op(self.value, other.value)
        result_dp = min(self.dec_places, other.dec_places)
        rounded = Decimal(raw).quantize(Decimal(f"1e-{result_dp}"), ROUND_HALF_UP)
        return SFNumber(rounded, _count_sig_figs(f"{rounded:.{result_dp}f}"), result_dp)

    # Multiplication / Division → sig-fig rule --------------------------
    def _muldiv(self, other: "SFNumber", op: Callable[[Decimal, Decimal], Decimal]) -> "SFNumber":
        raw = op(self.value, other.value)
        result_sf = min(self.sig_figs, other.sig_figs)
        rounded = self.__class__(raw, 0, 0)._round_to_sf(result_sf)  # type: ignore[arg-type]
        # Determine resulting decimal places *after* rounding so that later +/− knows what to do
        dp_after = max(0, -rounded.as_tuple().exponent)
        return SFNumber(rounded, result_sf, dp_after)

    # Operator dunders so we can recurse neatly -------------------------
    def __add__(self, other: "SFNumber") -> "SFNumber":
        return self._addsub(other, _op.add)

    def __sub__(self, other: "SFNumber") -> "SFNumber":
        return self._addsub(other, _op.sub)

    def __mul__(self, other: "SFNumber") -> "SFNumber":
        return self._muldiv(other, _op.mul)

    def __truediv__(self, other: "SFNumber") -> "SFNumber":
        return self._muldiv(other, _op.truediv)

    # Pretty printing ---------------------------------------------------
    def to_string(self) -> str:
        """Return a string retaining *exactly* the appropriate sig-fig/DPs."""
        if self.dec_places > 0:
            # We *must* show the fixed number of decimal places (even trailing 0s)
            return f"{self.value:.{self.dec_places}f}"
        # Otherwise use scientific notation if it helps avoid ambiguity
        if abs(self.value) >= Decimal("1e4") or (abs(self.value) != 0 and abs(self.value) < Decimal("1e-3")):
            # Use E-format, mantissa gets (sig_figs-1) decimals
            return f"{self.value:.{self.sig_figs - 1}E}"
        # Short plain format, but we must preserve trailing zeros from sig-fig rounding
        rounded = self._round_to_sf(self.sig_figs)
        return format(rounded, 'f')

# ──────────────────────────────────────────────────────────────────────────────
# Token analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_sig_figs(token: str) -> int:
    token = token.strip().lower()
    if token[0] in '+-':
        token = token[1:]
    # Scientific notation → everything before 'e' counts (except decimal point)
    if 'e' in token:
        mantissa = token.split('e')[0].replace('.', '')
        return len(mantissa)
    if '.' in token:
        # Any decimal → *all* digits except leading zeroes are significant
        digits = token.replace('.', '')
        return len(digits.lstrip('0'))
    # No decimal point → trailing zeros *not* significant
    return len(token.rstrip('0'))


def _count_dec_places(token: str) -> int:
    if 'e' in token.lower():
        # 2.3e4 style numbers are *exact* integers once scaled → 0 decimal places
        return 0
    if '.' in token:
        return len(token.split('.')[1])
    return 0

# ──────────────────────────────────────────────────────────────────────────────
# Expression parser using Python's AST (safe: only supports +, -, *, /, parens)
# ──────────────────────────────────────────────────────────────────────────────
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,  # ^ is *not* supported, but ** is via ast.Pow
)

_OPERATORS: dict[type[ast.AST], str] = {
    ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'
}

_BINARY_IMPL: dict[str, Callable[[SFNumber, SFNumber], SFNumber]] = {
    '+': SFNumber.__add__,
    '-': SFNumber.__sub__,
    '*': SFNumber.__mul__,
    '/': SFNumber.__truediv__,
}


def _compile_node(node: ast.AST, source_text: str) -> SFNumber:
    """Recursively convert an AST into nested :class:`SFNumber` operations."""
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError("Unsupported or unsafe expression component found.")

    match node:
        case ast.Expression(body=b):
            return _compile_node(b, source_text)
        case ast.Constant(value=v):
            # Re-extract token text (keeps trailing zeros which ast strips!)
            # We'll slice the source using the node's .col_offset where possible.
            # Fallback: str(v).
            if hasattr(node, 'col_offset') and node.col_offset is not None:
                # This is brittle but works well enough for tutoring examples
                m = TOKEN_RE.match(source_text[node.col_offset:])
                token = m.group('num') if m else str(v)
            else:
                token = str(v)
            return SFNumber.from_literal(token)
        case ast.UnaryOp(op=ast.UAdd(), operand=arg):
            return _compile_node(arg, source_text)
        case ast.UnaryOp(op=ast.USub(), operand=arg):
            base = _compile_node(arg, source_text)
            return SFNumber( -base.value, base.sig_figs, base.dec_places )
        case ast.BinOp(left=l, op=o, right=r):
            left_sf = _compile_node(l, source_text)
            right_sf = _compile_node(r, source_text)
            op_symbol = _OPERATORS[type(o)]
            return _BINARY_IMPL[op_symbol](left_sf, right_sf)
        case ast.Pow():
            raise ValueError("The ** operator isn't supported with sig-fig tracking.")
    raise AssertionError("Unhandled AST branch")

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def eval_with_sf(expr: str) -> str:
    """Evaluate *expr* and return a string already rounded for AP-Chem sig-figs.

    The expression may contain:
      * numbers in standard or scientific notation
      * +, -, *, / operators
      * parentheses for grouping
    """
    # Parse safely using python's ``ast`` module (no names, calls, etc.)
    tree = ast.parse(expr, mode='eval')
    result = _compile_node(tree, expr)
    return result.to_string()

if __name__ == '__main__':
    expr = ""
    eval_count = 1
    while True:
        expr = input(f'In [{eval_count}]: ')
        if expr == "exit":
            exit(0)
        print(f'Out [{eval_count}]: {eval_with_sf(expr)}')
        eval_count += 1
