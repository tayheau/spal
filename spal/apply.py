from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .hierarchy import Population
from .context import Context


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


@dataclass
class AnalysisResult:
    """Per-unit records ({**coords, 'value': ...}) plus the plan."""

    records: list[dict]
    context: Context

    @property
    def values(self) -> list[Any]:
        return [r["value"] for r in self.records]

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

    def aggregate(self, by, method: Any = "mean") -> "AnalysisResult":
        keys = (by,) if isinstance(by, str) else tuple(by)

        groups: dict[tuple, list[dict]] = {}
        for r in self.records:
            groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)

        out: list[dict] = []
        for gk, rows in groups.items():
            rec: dict = {}
            for c in set().union(*(r.keys() for r in rows)) - {"value", "n"}:
                vals = {r.get(c) for r in rows}      # keep coords constant in group
                if len(vals) == 1:
                    rec[c] = next(iter(vals))
            rec.update(dict(zip(keys, gk)))
            rec["value"] = _reduce([r["value"] for r in rows], method)
            rec["n"] = len(rows)
            out.append(rec)
        return AnalysisResult(out, self.context)

    def to(self, fmt: Any = "records"):
        if fmt in ("records", dict, list):
            return self.records
        if fmt in ("pandas", "df"):
            import pandas as pd
            return pd.DataFrame(self.records)
        if fmt in ("numpy", np.ndarray):
            return _stack(self.values)
        raise ValueError(f"unknown format {fmt!r}")


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
