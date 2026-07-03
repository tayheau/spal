from typing import Any, Callable, NamedTuple
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Record = dict[str, Any]

_MISSING: Any = object()

class Matrix(NamedTuple):
    values: npt.NDArray[np.float64]
    rows: list[tuple]
    cols: list[tuple]

class Dataset(NamedTuple):
    X: npt.NDArray[np.float64]
    y: npt.NDArray[Any]
    features: list[tuple]
    rows: list[Any]

def _is_scalar(v: Any) -> bool:
    return np.asarray(v).ndim == 0

def _label_eq(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    return bool(a == b)

def _yarr(vals: Sequence[Any], multi: bool) -> npt.NDArray[Any]:
    if multi:
        a = np.empty(len(vals), dtype=object)
        a[:] = list(vals)
        return a
    return np.asarray(vals)

def _ordered(keys, order):
    keys = list(keys)
    if order is not None:
        want = [k if isinstance(k, tuple) else (k,) for k in order]
        wantset = set(want)
        extra = [k for k in keys if k not in wantset]
        present = set(keys)
        return [k for k in want if k in present] + extra
    try:
        return sorted(keys)
    except TypeError:
        return keys

def _distinct(records: Sequence[Record], keys: Sequence[str], order=None) -> list[tuple]:
    seen = dict.fromkeys(tuple(r.get(k) for k in keys) for r in records)
    return _ordered(seen.keys(), order)

def _grid(
    records:list[Record], row_keys:str | Sequence[str],
    col_keys:str | Sequence[str], value:str = "value",
    fill:float =np.nan, reduce:Callable[[list[Any]], Any] | None = None,
    row_order:Sequence[Any] | None =None, col_order:Sequence[Any] | None =None,
) -> tuple[npt.NDArray[np.float64], list[tuple], list[tuple]]:
    rk = (row_keys,) if isinstance(row_keys, str) else tuple(row_keys)
    ck = (col_keys,) if isinstance(col_keys, str) else tuple(col_keys)

    if len(records) == 0: raise ValueError(f"No records to turn into a grid.")

    rows = _distinct(records, rk, row_order)
    cols = _distinct(records, ck, col_order)
    ri = {k: i for i, k in enumerate(rows)}
    cj = {k: j for j, k in enumerate(cols)}

    M = np.full((len(rows), len(cols)), fill, dtype=float)
    bucket: dict[tuple[int, int], list] | None = {} if reduce is not None else None
    for r in records:
        if not _is_scalar(val:=r.get(value)):
            raise ValueError( f"_grid needs a scalar {value!r}; got a vector. Reduce first")

        i = ri[tuple(r.get(c) for c in rk)]
        j = cj[tuple(r.get(c) for c in ck)]
        if bucket is None:
            M[i, j] = val
        else:
            bucket.setdefault((i, j), []).append(val)
    if bucket is not None:
        for (i, j), vs in bucket.items():
            M[i, j] = reduce(vs)
    return M, rows, cols

def _to_dataset(records, *, observation, features, label, value="value",
               align=None, fill=np.nan,
               feature_order=None, label_order=None) -> Dataset:
    """records -> Dataset(X, y, features, rows).

      observation=[coords]  one row per coord-tuple (e.g. unit); `value` SCALAR
                             -> X[row, feature].  individual = the neuron.
      observation="trial"   per-trial `value` VECTOR exploded, one row per trial;
                             columns = `features` (units), `label` = block
                             (condition).  individual = the trial (pop vector).

    align : "trial" only, REQUIRED — units can have different trial counts per
            block. "truncate" cuts to the block min, "pad" pads to max with
            `fill`. No default: you choose (the alignment a concat would hide).
    feature_order / label_order : explicit key orders (else sorted).
    """
    feat = (features,) if isinstance(features, str) else tuple(features)
    lab = (label,) if isinstance(label, str) else tuple(label)
    multi = len(lab) > 1

    def fkey(r): return tuple(r.get(c) for c in feat)
    def lkey(r):
        k = tuple(r.get(c) for c in lab)
        return k if multi else k[0]

    # ---- coord mode: X via _grid, y built alongside ----
    if observation != "trial":
        obs = (observation,) if isinstance(observation, str) else tuple(observation)
        X, rows, cols = _grid(records, obs, feat, value=value, fill=fill,
                              col_order=feature_order)          # scalar guard in _grid

        ylab: dict[tuple, Any] = {}
        for r in records:
            ok = tuple(r.get(c) for c in obs)
            lv = lkey(r)
            prev = ylab.get(ok, _MISSING)
            if prev is not _MISSING and not _label_eq(prev, lv):
                raise ValueError(f"label {label!r} varies within observation {ok};")
            ylab[ok] = lv
        y = _yarr([ylab[k] for k in rows], multi)
        return Dataset(X, y, cols, rows)

    # ---- trial mode: explode per-trial vectors into rows (not a _grid) ----
    if align not in ("truncate", "pad"):
        raise ValueError(
            '''observation='trial' needs align='truncate' or align='pad': units 
            can have different trial counts per block, and that must be explicit.''')
    if records and _is_scalar(records[0].get(value)):
        raise ValueError(
            '''observation='trial' expects a per-trial vector in `value` (e.g. 
             Csr.counts); got a scalar — use observation=<coords> instead.''')

    cols = _distinct(records, feat, feature_order)
    cidx = {k: i for i, k in enumerate(cols)}

    groups: dict[Any, list] = {}
    for r in records:
        groups.setdefault(tuple(r.get(c) for c in lab), []).append(r)
    blocks_order = _ordered(groups.keys(), label_order)

    blocks, ys, rows = [], [], []
    for tk in blocks_order:
        grp = groups[tk]
        lk = tk if multi else tk[0]
        lens = [len(np.asarray(r.get(value))) for r in grp]
        T = min(lens) if align == "truncate" else max(lens)
        block = np.full((T, len(cols)), fill, dtype=float)

        seen: set[int] = set()
        for r in grp:
            j = cidx[fkey(r)]
            if j in seen:
                raise ValueError(f"two records share same feature {cols[j]} in block {lk!r}.\ntrial mode cannot merge them : deduplicate or pool upstream.")
            seen.add(j)
            v = np.asarray(r.get(value), dtype=float)
            if align == "truncate":
                block[:, j] = v[:T]
            else:
                block[:len(v), j] = v
        blocks.append(block)
        ys += [lk] * T
        rows += [(lk, t) for t in range(T)]

    X = np.vstack(blocks) if blocks else np.empty((0, len(cols)))
    return Dataset(X, _yarr(ys, multi), cols, rows)
