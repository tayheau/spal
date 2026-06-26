from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .hierarchy import Population
from .context import Context
from .sparklines import spark

RESERVED = frozenset({"value"})

def _stack(values: list):
    try:
        return np.stack([np.asarray(v) for v in values])
    except ValueError:
        return np.array(values, dtype=object)  # ragged -> object array

def _reduce(values: list, method: Any):
    if callable(method):
        return method(values)
    if method in ("stack", None):
        return _stack(values)
    fns = {"mean": np.nanmean, "sum": np.nansum,
           "median": np.nanmedian, "std": np.nanstd}
    if method not in fns:
        raise ValueError(f"unknown aggregation method {method!r}")
    return fns[method](_stack(values), axis=0)

def _hashable(v):
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, np.ndarray):
        return tuple(v.flat)
    return v

@dataclass
class AnalysisResult:
    """Per-unit records ({**coords, 'value': ...}) plus the plan."""

    records: list[dict[str, Any]]
    context: Context

    @property
    def values(self) -> list[Any]:
        return [r["value"] for r in self.records]

    @property
    def coord_keys(self) -> set[str]:
        if not self.records: return set()
        return self.records[0].keys() - RESERVED

    def __len__(self) -> int:
        return len(self.records)

    def where(self, **conditions) -> "AnalysisResult":
        def ok(r):
            for k, v in conditions.items():
                g = r.get(k)
                if callable(v):
                    if not v(g):
                        return False
                elif isinstance(v, (list, tuple, set)):
                    if g not in v:
                        return False
                elif g != v:
                    return False
            return True

        return AnalysisResult([r for r in self.records if ok(r)], self.context)

    def reduce_by(self, by, fn: Callable[[list[dict]], Any]) -> "AnalysisResult":
        keys = (by,) if isinstance(by, str) else tuple(by)
        if (invalid := set(keys) - self.coord_keys):
            raise ValueError(f"Unkown coord keys : {invalid}")
        groups: dict[tuple, list[dict]] = {}
        for r in self.records:
            groups.setdefault(tuple(_hashable(r.get(k)) for k in keys), []).append(r)
        out = []
        for gk, rows in groups.items():
            rec: dict = {}
            for c in self.coord_keys:
                vals = {_hashable(r.get(c)) for r in rows}
                if len(vals) == 1:
                    rec[c] = next(iter(vals))
            rec.update(dict(zip(keys, gk)))
            res = fn(rows)
            rec.update(res if isinstance(res, dict) else {"value": res})
            out.append(rec)
        return AnalysisResult(out, self.context)

    def aggregate(self, by:str|Sequence[str]|None = None,
                  method: Any = "mean",) -> "AnalysisResult":
        return self.reduce_by([] if by is None else by,
                              lambda rows: _reduce([r["value"] for r in rows], method))

    def to(self, fmt: Any = "records"):
        if fmt in ("records", dict, list):
            return self.records
        if fmt in ("pandas", "df"):
            import pandas as pd
            return pd.DataFrame(self.records)
        if fmt in ("numpy", np.ndarray):
            return _stack(self.values)
        raise ValueError(f"unknown format {fmt!r}")

    def get_unique_coord_values(self, name: str, exclude_none: bool = True):
        coords = self.coord_keys
        if not name in coords:
            raise ValueError(f"{name!r} is not a valid coord: {coords}.")
        return {c for r in self.records if (c:=r.get(name)) is not None or not exclude_none}

    def __repr__(self) -> str:
        plan = " → ".join(type(op).__name__ for op in self.context.ops) or "∅"
        n = len(self.records)
        if n == 0:
            return f"AnalysisResult(empty | {plan})"

        coords = self.coord_keys
        varying = [(k, c) for k in coords
                   if (c := len({_hashable(r.get(k)) for r in self.records})) > 1]
        dims = ", ".join(f"{k}×{c}" for k, c in varying)

        vals = [r["value"] for r in self.records]
        scalar = all(np.ndim(v) == 0 for v in vals)
        if scalar:
            a = np.asarray(vals, float)
            vsum = f"value∈[{a.min():.3g}, {a.max():.3g}]"
        else:
            vsum = f"value: {np.asarray(vals[0]).shape} arrays"

        parts = [f"{n} records", dims, vsum, plan]
        base = "AnalysisResult(" + " | ".join(p for p in parts if p) + ")"

        line = None
        if scalar and len(varying) == 1:
            k = varying[0][0]
            order = sorted(range(n), key=lambda j: self.records[j].get(k))
            line = spark(a[order])
        elif not scalar and n == 1:
            line = spark(np.asarray(vals[0]))
        return base if line is None else base + "\n" + line

    # def _repr_html_(self):

def apply(
    population: Population,
    fn: Callable,
    ctx: Context | None = None,
) -> AnalysisResult:

    ctx = ctx or Context()  # empty plan -> full population, no windowing

    records = [
        {**uc.coords, "value": fn(uc)}
        for uc in ctx.stream(population)
    ]

    return AnalysisResult(records=records, context=ctx)
