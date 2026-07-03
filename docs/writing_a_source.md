# Writing a SpikeSource

A `SpikeSource` is the one adapter you write to connect your data to spal. Once
your data speaks this small protocol, every spal analysis runs on it — the same
metric someone wrote for their recordings runs on yours, unchanged.

This is the [PyTorch `Dataset`](https://pytorch.org/docs/stable/data.html) idea
for spike trains: concentrate all the per-dataset glue in one thin adapter, and
the analysis on top becomes portable. Paste the analysis, supply the source, run.

## The protocol

A `SpikeSource` is any object that exposes two things:

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class SpikeSource(Protocol):
    @property
    def unit_ids(self) -> list[str]:
        """Ids of the units this source can serve."""

    def spikes(self, unit_id: str) -> np.ndarray:
        """Return spike timestamps in seconds (sorted ascending)."""
```

That's the whole contract:

- **`unit_ids`** — a property listing the identifiers of the units your source
  holds. spal iterates these to build the recording's roster.
- **`spikes(unit_id)`** — given an id, return that unit's spike times as a
  1-D NumPy array of seconds, sorted ascending.

Because it's a
[`runtime_checkable`](https://docs.python.org/3/library/typing.html#typing.runtime_checkable)
`Protocol`, you don't subclass anything. Any object with a matching `unit_ids`
property and `spikes` method *is* a `SpikeSource` — spal checks structurally, not
by inheritance. Your class doesn't import from spal at all.

!!! note "Why `spikes()` is a method, not stored data"
    Returning spikes through a method — rather than exposing a dict of all
    trains — is what lets a source stay lazy. A source backed by a file can read
    one unit's spikes from disk on demand and let them be garbage-collected after
    use, so peak memory doesn't grow with the recording. spal streams units
    through the analysis, and a lazy `spikes()` is what makes that pay off.

## A minimal source

Here is the smallest useful source: a fixed roster backed by an in-memory dict.

```python
import numpy as np

class DictSpikeSource:
    """Wrap {unit_id: spike_times} — the simplest possible source."""

    def __init__(self, trains: dict[str, np.ndarray]):
        self._trains = {k: np.sort(np.asarray(v, dtype=float)) for k, v in trains.items()}

    @property
    def unit_ids(self) -> list[str]:
        return list(self._trains)

    def spikes(self, unit_id: str) -> np.ndarray:
        return self._trains[unit_id]
```

That's a complete, valid `SpikeSource`. Plug it in:

```python
from spal.hierarchy import Population, Subject, Recording
from spal.apply import apply

src = DictSpikeSource({
    "a": [0.10, 0.52, 0.91, 1.20],
    "b": [0.33, 0.48, 1.05],
})
rec = Recording.from_source("rec0", src)
pop = Population("demo", [Subject("m1", [rec])])

res = apply(pop, lambda uc: len(uc.spikes))   # spike count per unit
# AnalysisResult: one record per unit, {unit_id, ..., value}
```

`Recording.from_source` reads `src.unit_ids` to build one `Unit` per id, and each
unit's `spikes` are served through your `spikes()` method when a metric asks for
them.

## Checking your source

Since the protocol is `runtime_checkable`, you can assert conformance directly:

```python
from spal.source import SpikeSource

assert isinstance(src, SpikeSource)   # True if unit_ids + spikes are present
```

This checks the *shape* (the attributes exist), not the *behaviour*. Two things
the check can't verify but your source must honour:

- `spikes()` returns a **1-D array of floats in seconds**, not sample indices or
  a 2-D array.
- the timestamps are **sorted ascending** — windowing relies on it (spal uses
  `searchsorted` to align spikes to events, which assumes sorted input).

If your raw data is in samples, divide by the sampling rate before returning. If
it isn't sorted, `np.sort` it once in your source rather than leaving spal to
discover the problem downstream.

## Attaching metadata

A source serves *spikes*. Anything else about a unit — depth, region, channel,
cluster quality — is **metadata**, and it's attached when you build the
recording, not inside the source:

```python
rec = Recording.from_source(
    "rec0", src,
    unit_metadata={
        "a": {"depth": 120, "region": "V1"},
        "b": {"depth": 340, "region": "M1"},
    },
)
```

These keys become coordinates that propagate into every result, so you can later
`aggregate(by="region")` or filter `where(depth=...)`. Keeping metadata out of the
source keeps the source's job small: it answers *where are the spikes*, nothing
more.

## A lazy, file-backed source

The point of the protocol is that a source can read from anywhere. Here's the
shape of a source that reads spikes from disk on demand — one unit at a time,
never holding the whole recording in memory:

```python
import numpy as np

class MemmapSpikeSource:
    """Reads each unit's spikes from a memory-mapped file on demand."""

    def __init__(self, index: dict[str, tuple[int, int]], path: str, fs: float):
        # index maps unit_id -> (start, stop) offsets into a flat samples file
        self._index = index
        self._data = np.memmap(path, dtype=np.int64, mode="r")
        self._fs = fs                                   # sampling rate, Hz

    @property
    def unit_ids(self) -> list[str]:
        return list(self._index)

    def spikes(self, unit_id: str) -> np.ndarray:
        start, stop = self._index[unit_id]
        samples = np.asarray(self._data[start:stop])    # only this unit is read
        return np.sort(samples / self._fs)              # samples -> seconds
```

Only the requested unit's slice is touched per call. Combined with spal's
streaming execution, peak memory stays roughly flat regardless of how long the
recording is — the whole cohort is never materialized at once.

## Where to go from here

- The synthetic [`RandomSpikeSource`](../api/source.md) that ships with spal is a
  worked example — a fixed roster of Poisson units, cached on first use, useful
  for tests and examples.
- Adapters for common formats (SpikeInterface, NWB) are on the roadmap; until
  they land, a thin adapter like the ones above is a few lines.
- Once your source is in place, see [Exporting for ML](export.md) to turn a
  result into a design matrix for ML librairies.
