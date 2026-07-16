from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, overload, Literal
from typing_extensions import override
from collections.abc import Sequence

import numpy as np

from .export import Matrix, _grid, Dataset, _to_dataset
from .hierarchy import Population
from .context import Context
from .sparklines import spark

def _stack(values: list):
    try:
        return np.stack([np.asarray(v) for v in values])
    except ValueError:
        return np.array(values, dtype=object)  # ragged -> object array

def _reduce(values: list, method: Any):
    if callable(method):
        return method(values)
    if method in ("stack", None):
        return _stack(values).tolist()
    fns = {"mean": np.nanmean, "sum": np.nansum,
           "median": np.nanmedian, "std": np.nanstd}
    if method not in fns:
        raise ValueError(f"unknown aggregation method {method!r}")
    return fns[method](_stack(values), axis=0).tolist()

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
    measures: frozenset[str] = frozenset({"value"})

    def get_values(self, keys: str | Sequence[str] | None = None) -> dict[str, list[Any]] | list[Any]:
        """
        Return the values in record order of the `keys` (coords and measures)
        """
        _keys: set[str] | None = {keys,} if isinstance(keys, str) else set(keys) if keys is not None else None
        if _keys is not None and (_r:= _keys - self.measures.union(self.coord_keys)):
            raise KeyError(f"Unknown key: {_r!r}. Valid ones are {self.measures.union(self.coord_keys)!r}")
        mk = list(self.measures) if _keys is None else list(_keys)
        return [r[mk[0]] for r in self.records] if len(mk) == 1 else {k:[r.get(k) for r in self.records] for k in mk}

    @property
    def coord_keys(self) -> set[str]:
        if not self.records: return set()
        return self.records[0].keys() - self.measures

    def __len__(self) -> int:
        return len(self.records)

    def where(self, **conditions) -> "AnalysisResult":
        if _e:= set(conditions.keys()) - self.measures.union(self.coord_keys):
            raise KeyError(f"Unknown key(s): {_e!r}.")
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

        return AnalysisResult([r for r in self.records if ok(r)], self.context, self.measures)

    def aggregate_using(self, by: str | Sequence[str],
                        fn: Callable[[dict[str, list]], Any]) -> "AnalysisResult":
        keys = (by,) if isinstance(by, str) else tuple(by)
        if (invalid := set(keys) - self.coord_keys):
            raise ValueError(f"Unknown coord keys : {invalid}")
        groups: dict[tuple, list[dict]] = {}
        for r in self.records:
            groups.setdefault(tuple(_hashable(r.get(k)) for k in keys), []).append(r)
        out, out_measures = [], None
        for gk, rows in groups.items():
            rec: dict = {}
            for c in self.coord_keys:
                vals = {_hashable(r.get(c)) for r in rows}
                if len(vals) == 1: rec[c] = next(iter(vals))
            rec.update(dict(zip(keys, gk)))
            cols = {m: [r.get(m) for r in rows] for m in self.measures}
            res = fn(cols)
            contributed = res if isinstance(res, dict) else {"value": res}
            if out_measures is None:
                out_measures = frozenset(contributed)          # <- les clés que fn a produites
            rec.update(contributed)
            out.append(rec)
        return AnalysisResult(out, self.context, out_measures or frozenset({"value"}))

    def aggregate(self, by:str|Sequence[str]|None = None,
                  method:Literal["mean", "sum", "median", "std", "stack"]= "mean",
                  measure: str | Sequence[str] | None = None) -> "AnalysisResult":
        m = self.measures if measure is None else measure
        return self.aggregate_using([] if by is None else by,
                                    lambda cols: {k:_reduce(cols[k], method) for k in m})

    def to(self, fmt: Any = "records"):
        if fmt in ("records", dict, list):
            return self.records
        if fmt in ("pandas", "df"):
            import pandas as pd
            return pd.DataFrame(self.records)
        if fmt in ("numpy", np.ndarray):
            return _stack(self.get_values)
        raise ValueError(f"unknown format {fmt!r}")

    def get_unique_coord_values(self, name: str, exclude_none: bool = True):
        coords = self.coord_keys
        if not name in coords:
            raise ValueError(f"{name!r} is not a valid coord: {coords}.")
        return {c for r in self.records if (c:=r.get(name)) is not None or not exclude_none}

    @override
    def __repr__(self) -> str:
        plan = " → ".join(type(op).__name__ for op in self.context.ops) or "∅"
        n = len(self.records)
        if n == 0:
            return f"AnalysisResult(empty | {plan})"

        coords = self.coord_keys
        varying = [(k, c) for k in coords
                   if (c := len({_hashable(r.get(k)) for r in self.records})) > 1]
        dims = ", ".join(f"{k}×{c}" for k, c in varying)

        def _summ(mk):
            vals = [r.get(mk) for r in self.records]
            if all(np.ndim(v) == 0 for v in vals):
                a = np.asarray(vals, float)
                return f"{mk}∈[{np.nanmin(a):.3g}, {np.nanmax(a):.3g}]"
            return f"{mk}: {np.asarray(vals[0]).shape} arrays"

        measure_summary = ", ".join(_summ(mk) for mk in self.measures)
        parts = [f"{n} records", dims, measure_summary, plan]
        base = "AnalysisResult(" + " | ".join(p for p in parts if p) + ")"
        
        # line = None
        # if n == 1:
        #     values = self.get_values(list(self.measures))
        #     for k, v in values.items():
        #         if np.ndim(v) > 0:
        #             line = f"{k}: {spark

        return base

        # sparkline only for a SINGLE scalar measure — ambiguous otherwise
        # line = None
        # if len(self.measures) == 1:
        #     mk = next(iter(self.measures))
        #     vals = [r.get(mk) for r in self.records]
        #     scalar = all(np.ndim(v) == 0 for v in vals)
        #     if scalar and len(varying) == 1:
        #         a = np.asarray(vals, float)
        #         k = varying[0][0]
        #         order = sorted(range(n), key=lambda j: self.records[j].get(k))
        #         line =f"{k}: {spark(a[order])}"
        #     elif not scalar and n == 1:
        #         line = spark(np.asarray(vals[0]))
        # return base if line is None else base + "\n" + line

    def to_matrix(self, rows:str | Sequence[str], cols: str | Sequence[str], value: str="value", fill:float=np.nan,
                  reduce: Callable[[list[Any]], Any] | None=None, row_order: Sequence[Any] | None=None,
                  col_order:Sequence[Any] | None=None):
        M, R, C = _grid(self.records, rows, cols, value=value, fill=fill, reduce=reduce,
                        row_order=row_order, col_order=col_order)
        return Matrix(M, R, C)

    @overload
    def to_dataset(self, *, observation: Literal["trial"],
                   features: str | Sequence[str], label: str | Sequence[str],
                   align: Literal["truncate", "pad"],
                   value: str = ..., fill: float = ...,
                   feature_order: Sequence[Any] | None = ...,
                   label_order: Sequence[Any] | None = ...) -> Dataset: ...
    @overload
    def to_dataset(self, *, observation: str | Sequence[str],
                   features: str | Sequence[str], label: str | Sequence[str],
                   value: str = ..., fill: float = ...,
                   feature_order: Sequence[Any] | None = ...,
                   label_order: Sequence[Any] | None = ...) -> Dataset: ...

    def to_dataset(self, *, observation: str | Sequence[str],
                   features: str | Sequence[str], label: str | Sequence[str],
                   align: Literal["truncate", "pad"] | None = None,
                   value: str = "value", fill: float = np.nan,
                   feature_order: Sequence[Any] | None = None,
                   label_order: Sequence[Any] | None = None) -> Dataset:
        return _to_dataset(self.records, observation=observation, features=features,
                           label=label, align=align, value=value, fill=fill,
                           feature_order=feature_order, label_order=label_order)

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
