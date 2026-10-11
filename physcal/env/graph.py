"""Formula graph: quantities, sympy relations, derivation closure, and chain sampling.

A family is one coherent physical scenario (a "world"). Its formulas are sympy relations over the
scenario's named quantities. A problem is a linear chain of formula applications: step i consumes the
output of step i-1 plus given quantities, and the last step produces the target.

The chain's depth is honest: the target's minimal derivation depth from the givens, computed over *all*
formulas of the family, must equal the chain length (no shortcuts), and no given may be derivable from the
other givens (no over-determined problems).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import sympy as sp

from physcal.env.units import UNIT_MENUS, quantity, to_si

# Numerical settings for the structural independence check (not experiment thresholds).
REFERENCE_WORLD_SEED = 0  # the generic rank is evaluated at one fixed random world
REFERENCE_WORLD_TRIES = 1000
RANK_RTOL = 1e-8  # singular values below this fraction of the largest count as zero


@dataclass(frozen=True)
class Quantity:
    symbol: str
    desc: str
    si_unit: str
    # Log-uniform sampling range in SI. Only needed for basis quantities and distractors.
    range_si: tuple[float, float] | None = None
    # Plausibility bounds in SI: worlds with this quantity outside the bounds are resampled.
    bounds_si: tuple[float, float] | None = None

    @property
    def units(self) -> tuple[str, ...]:
        return UNIT_MENUS[self.si_unit]


@dataclass(frozen=True)
class Solved:
    """One solve direction of a formula: `output = expr(args)`."""

    output: str
    expr: sp.Expr
    args: tuple[str, ...]
    func: Callable[..., Any]

    def __call__(self, values: dict[str, Any]) -> Any:
        return self.func(*(values[a] for a in self.args))


def _positive_branch(solutions: list[sp.Expr]) -> list[sp.Expr]:
    """All quantities are positive magnitudes: drop branches sympy can tell are not positive."""
    keep = [s for s in solutions if not (s.is_negative or s.is_zero or s.is_nonpositive)]
    if len(keep) > 1:
        keep = [s for s in keep if not s.could_extract_minus_sign()]
    return keep


class Formula:
    """A sympy relation `lhs = rhs` and the symbols it may be solved for."""

    def __init__(self, name: str, relation: str, solve_for: str | Sequence[str]):
        self.name = name
        self.relation = relation
        self.solve_for: tuple[str, ...] = tuple(solve_for.split() if isinstance(solve_for, str) else solve_for)
        self.eq: sp.Eq | None = None
        self.variables: tuple[str, ...] = ()
        self.constants: tuple[str, ...] = ()
        self.solved: dict[str, Solved] = {}

    def bind(self, table: dict[str, sp.Symbol], constants: Iterable[str]) -> None:
        constants = set(constants)
        lhs, rhs = self.relation.split("=")
        self.eq = sp.Eq(sp.parse_expr(lhs, local_dict=table), sp.parse_expr(rhs, local_dict=table))
        names = sorted(str(s) for s in self.eq.free_symbols)
        unknown = [n for n in names if n not in table]
        if unknown:
            raise ValueError(f"formula {self.name}: unknown symbols {unknown}")
        self.variables = tuple(n for n in names if n not in constants)
        self.constants = tuple(n for n in names if n in constants)
        for out in self.solve_for:
            if out not in self.variables:
                raise ValueError(f"formula {self.name}: cannot solve for {out!r}, not a variable")
            sols = _positive_branch(sp.solve(self.eq, table[out]))
            if len(sols) != 1:
                raise ValueError(f"formula {self.name}: solving for {out} is not unique: {sols}")
            expr = sols[0]
            args = tuple(sorted(str(s) for s in expr.free_symbols))
            func = sp.lambdify([table[a] for a in args], expr, modules="numpy")
            self.solved[out] = Solved(out, expr, args, func)

    def inputs(self, output: str) -> tuple[str, ...]:
        """Non-constant quantities needed to compute `output` with this formula."""
        return tuple(v for v in self.variables if v != output)

    def sides(self, values: dict[str, Any]) -> tuple[Any, Any]:
        """Evaluate lhs and rhs numerically (floats or pint quantities)."""
        out = []
        for side in (self.eq.lhs, self.eq.rhs):
            args = sorted(side.free_symbols, key=str)
            out.append(sp.lambdify(args, side, modules="numpy")(*(values[str(s)] for s in args)))
        return out[0], out[1]

    def __repr__(self) -> str:
        return f"Formula({self.name}: {self.relation})"


@dataclass(frozen=True)
class Chain:
    """Linear derivation: `steps[i] = (formula_name, output)` in evaluation order."""

    steps: tuple[tuple[str, str], ...]
    givens: tuple[str, ...]

    @property
    def target(self) -> str:
        return self.steps[-1][1]

    @property
    def intermediates(self) -> tuple[str, ...]:
        return tuple(out for _, out in self.steps[:-1])

    @property
    def depth(self) -> int:
        return len(self.steps)


class Family:
    """One scenario: quantities, constants, formulas, a world sampler and distractors."""

    def __init__(
        self,
        name: str,
        intros: Sequence[str],
        quantities: Sequence[Quantity],
        formulas: Sequence[Formula],
        basis: Sequence[str],
        sample_basis: Callable[[np.random.Generator, dict[str, Quantity]], dict[str, float]] | None = None,
        constants: dict[str, tuple[float, str]] | None = None,
        distractors: Sequence[Quantity] = (),
    ):
        self.name = name
        self.intros = tuple(intros)
        self.quantities = {q.symbol: q for q in quantities}
        self.constants = dict(constants or {})
        self.si_constants = {k: to_si(v, u) for k, (v, u) in self.constants.items()}
        self.basis = tuple(basis)
        self.distractors = {q.symbol: q for q in distractors}
        self._sample_basis = sample_basis or _log_uniform_basis(self.basis)
        clash = set(self.quantities) & (set(self.constants) | set(self.distractors))
        if clash:
            raise ValueError(f"{name}: symbol clash {clash}")
        table = {s: sp.Symbol(s, positive=True) for s in [*self.quantities, *self.constants]}
        self.symbols = table
        for f in formulas:
            f.bind(table, self.constants)
        self.formulas = tuple(formulas)
        self._by_name = {f.name: f for f in self.formulas}
        self.derivable = tuple(sorted({o for f in self.formulas for o in f.solve_for}))
        self.world_plan = self._plan_world()
        self._chain_cache: dict[int, dict[str, tuple[Chain, ...]]] = {}
        self._levels_cache: dict[frozenset[str], dict[str, int]] = {}
        self._independent_cache: dict[frozenset[str], bool] = {}

    def formula(self, name: str) -> Formula:
        return self._by_name[name]

    # ---------------------------------------------------------------- derivation closure
    def derivation_levels(self, known: Iterable[str]) -> dict[str, int]:
        """Minimal derivation depth of every quantity reachable from `known` (level 0)."""
        level = {q: 0 for q in known}
        k = 0
        while True:
            k += 1
            new: dict[str, int] = {}
            for f in self.formulas:
                for out in f.solve_for:
                    if out in level or out in new:
                        continue
                    if all(i in level for i in f.inputs(out)):
                        new[out] = k
            if not new:
                return level
            level.update(new)

    def _plan_world(self) -> tuple[tuple[str, str], ...]:
        known = set(self.basis)
        plan: list[tuple[str, str]] = []
        progress = True
        while progress:
            progress = False
            for f in self.formulas:
                for out in f.solve_for:
                    if out not in known and all(i in known for i in f.inputs(out)):
                        plan.append((f.name, out))
                        known.add(out)
                        progress = True
        missing = set(self.quantities) - known
        if missing:
            raise ValueError(f"{self.name}: basis {self.basis} does not determine {sorted(missing)}")
        return tuple(plan)

    # ---------------------------------------------------------------- worlds
    def sample_world(self, rng: np.random.Generator, max_tries: int) -> dict[str, float]:
        """A consistent assignment of every quantity, in SI, derived from sampled basis values."""
        for _ in range(max_tries):
            values = dict(self._sample_basis(rng, self.quantities))
            env = {**values, **self.si_constants}
            with np.errstate(invalid="ignore", divide="ignore"):
                for name, out in self.world_plan:
                    env[out] = float(self.formula(name).solved[out](env))
            world = {q: env[q] for q in self.quantities}
            if all(math.isfinite(v) and v > 0 for v in world.values()) and self.within_bounds(world):
                return world
        raise RuntimeError(f"{self.name}: no valid world after {max_tries} tries")

    def within_bounds(self, values: dict[str, float]) -> bool:
        for sym, v in values.items():
            bounds = self.quantities[sym].bounds_si
            if bounds is not None and not bounds[0] <= v <= bounds[1]:
                return False
        return True

    # ---------------------------------------------------------------- chains
    def chains(self, depth: int) -> dict[str, tuple[Chain, ...]]:
        """Every valid depth-`depth` chain, grouped by target. Exhaustive and deterministic; cached."""
        if depth not in self._chain_cache:
            found: dict[str, list[Chain]] = {}
            for target in self.derivable:
                for chain in self._grow(target, depth, (target,), frozenset(), frozenset(), ()):
                    if self.is_valid_chain(chain):
                        found.setdefault(target, []).append(chain)
            self._chain_cache[depth] = {t: tuple(sorted(set(c), key=repr)) for t, c in sorted(found.items())}
        return self._chain_cache[depth]

    def _grow(self, current, level, outputs, used, givens, steps):
        """Depth-first backward growth of linear chains ending in `outputs[0]`."""
        for f in self.formulas:
            if current not in f.solve_for or f.name in used:
                continue
            inputs = f.inputs(current)
            if any(i in outputs for i in inputs):
                continue
            here = steps + ((f.name, current),)
            if level == 1:
                yield Chain(steps=tuple(reversed(here)), givens=tuple(sorted(givens | set(inputs))))
                continue
            for nxt in inputs:
                if nxt not in self.derivable:
                    continue
                g = givens | {i for i in inputs if i != nxt}
                if nxt in g or g & set(outputs):
                    continue
                yield from self._grow(nxt, level - 1, outputs + (nxt,), used | {f.name}, g, here)

    def is_valid_chain(self, chain: Chain) -> bool:
        """No shortcut (step i has minimal depth i) and algebraically independent givens."""
        levels = self._levels(frozenset(chain.givens))
        for i, (_, out) in enumerate(chain.steps, start=1):
            if levels.get(out) != i:
                return False
        return self.independent(frozenset(chain.givens))

    def _levels(self, known: frozenset[str]) -> dict[str, int]:
        if known not in self._levels_cache:
            self._levels_cache[known] = self.derivation_levels(known)
        return self._levels_cache[known]

    def independent(self, givens: frozenset[str]) -> bool:
        """True if no relation ties the givens together, i.e. the problem is not over-determined.

        Every quantity is a function of the basis, so the givens are independent iff the Jacobian
        d ln(givens) / d ln(basis) has full row rank. This also catches constraints that need
        simultaneous equations, which the one-formula-at-a-time closure cannot see.
        """
        if givens not in self._independent_cache:
            if len(givens) > len(self.basis):
                ok = False
            else:
                jac = np.array([self._log_jacobian[g] for g in sorted(givens)])
                sv = np.linalg.svd(jac, compute_uv=False)
                ok = bool(sv.min() > RANK_RTOL * sv.max())
            self._independent_cache[givens] = ok
        return self._independent_cache[givens]

    @cached_property
    def _log_jacobian(self) -> dict[str, np.ndarray]:
        """d ln q / d ln b for every quantity q and basis quantity b, at a fixed reference world."""
        exprs: dict[str, sp.Expr] = {b: self.symbols[b] for b in self.basis}
        for name, out in self.world_plan:
            solved = self.formula(name).solved[out]
            exprs[out] = solved.expr.subs({self.symbols[a]: exprs[a] for a in solved.args if a in exprs})
        world = self.sample_world(np.random.default_rng(REFERENCE_WORLD_SEED), REFERENCE_WORLD_TRIES)
        point = {self.symbols[k]: v for k, v in {**world, **self.si_constants}.items()}
        return {
            q: np.array(
                [float((sp.diff(e, self.symbols[b]) * self.symbols[b] / e).evalf(30, subs=point)) for b in self.basis]
            )
            for q, e in exprs.items()
        }

    def sample_chain(self, depth: int, rng: np.random.Generator) -> Chain:
        """Uniform over targets that admit a depth-`depth` chain, then uniform over that target's chains."""
        by_target = self.chains(depth)
        if not by_target:
            raise RuntimeError(f"{self.name}: no valid depth-{depth} chain exists")
        targets = list(by_target)
        options = by_target[targets[rng.integers(len(targets))]]
        return options[rng.integers(len(options))]

    def evaluate_chain(self, chain: Chain, givens_si: dict[str, float]) -> dict[str, float]:
        """Run the chain on plain SI floats. Returns every step output."""
        env: dict[str, Any] = {**givens_si, **self.si_constants}
        with np.errstate(invalid="ignore", divide="ignore"):
            for name, out in chain.steps:
                env[out] = float(self.formula(name).solved[out](env))
        return {out: env[out] for _, out in chain.steps}

    def evaluate_chain_with_units(self, chain: Chain, givens_si: dict[str, float]) -> dict[str, float]:
        """Run the chain with pint quantities, checking every output's dimensions. Returns SI magnitudes."""
        env: dict[str, Any] = {g: quantity(v, self.quantities[g].si_unit) for g, v in givens_si.items()}
        env.update({k: quantity(v, u) for k, (v, u) in self.constants.items()})
        result: dict[str, float] = {}
        for name, out in chain.steps:
            with np.errstate(invalid="ignore", divide="ignore"):
                val = self.formula(name).solved[out](env)
            if not hasattr(val, "to_base_units"):  # pint collapses a dimensionless result to a float
                val = quantity(float(val), "")
            val = val.to_base_units()
            expected = quantity(1.0, self.quantities[out].si_unit).dimensionality
            if val.dimensionality != expected:
                raise ValueError(f"{name} -> {out}: got {val.dimensionality}, expected {expected}")
            env[out] = val
            result[out] = float(val.magnitude)
        return result

    def chain_expression(self, chain: Chain) -> sp.Expr:
        """The target as one sympy expression in the givens and constants."""
        expr: sp.Expr = self.symbols[chain.target]
        for name, out in reversed(chain.steps):
            expr = expr.subs(self.symbols[out], self.formula(name).solved[out].expr)
        return expr

    def condition_number(self, chain: Chain, givens_si: dict[str, float], rel_step: float) -> float:
        """max over givens and outputs of |d ln(output) / d ln(given)|, by finite differences."""
        base = self.evaluate_chain(chain, givens_si)
        kappa = 0.0
        for g in chain.givens:
            bumped = dict(givens_si)
            bumped[g] *= 1.0 + rel_step
            out = self.evaluate_chain(chain, bumped)
            for k, v in base.items():
                kappa = max(kappa, abs((out[k] - v) / v) / rel_step)
        return kappa

    @cached_property
    def max_depth(self) -> int:
        """Longest minimal derivation depth from the basis (an upper-bound sanity figure)."""
        return max(self.derivation_levels(self.basis).values())


def _log_uniform_basis(basis: Sequence[str]):
    def sample(rng: np.random.Generator, quantities: dict[str, Quantity]) -> dict[str, float]:
        return {b: log_uniform(rng, *quantities[b].range_si) for b in basis}

    return sample


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
