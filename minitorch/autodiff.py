from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


def central_difference(f: Any, vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
        f: arbitrary function from n-scalar args to one value
        vals: n-float values x1.0 ... x(n-1)
        arg: the number `i` of the arg to compute the derivative
        epsilon: a small constant

    Returns:
        An approximation of f'(x1.0, ..., x(n-1))
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable["Variable"]: ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]: ...


def topological_sort(variable: "Variable") -> Iterable["Variable"]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order: List["Variable"] = []
    seen = set()

    def visit(var: "Variable") -> None:
        if var.unique_id in seen or var.is_constant():
            return
        for v in var.parents:
            if not v.is_leaf():
                visit(v)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: "Variable", deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaves."""
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in queue:
        d = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for v, d_output in var.chain_rule(d):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] += d_output


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given values if they need to be used during backpropagation."""
        if not self.no_grad:
            self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
