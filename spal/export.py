from typing import Any, Callable, NamedTuple
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Record = dict[str, Any]

class Matrix(NamedTuple):
    values: npt.NDArray[np.float64]
    rows: list[tuple]
    cols: list[tuple]

def _is_scalar(v: Any) -> bool:
    return np.asarray(v).ndim == 0

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
