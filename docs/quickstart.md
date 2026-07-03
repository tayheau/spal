# Quickstart

This walks the whole path — from spikes to a design matrix — in one short script,
using spal's synthetic source so it runs with no data of your own.

## The pipeline in four steps

```python
import numpy as np
from spal.hierarchy import Population, Subject, Recording
from spal.context   import ContextBuilder
from spal.ops       import GroupOp, WindowOp
from spal.source    import RandomSpikeSource
from spal.stimulus  import StimulusTable
from spal.apply     import apply

# 1. plug your data in behind the SpikeSource protocol (synthetic here)
onsets = np.arange(1.0, 100.0, 2.0)
stim   = StimulusTable(onsets, intensity=np.random.choice([0, 40], len(onsets)))
src    = RandomSpikeSource(onsets, n_units=32, seed=0)
rec    = Recording.from_source("rec0", src, stim)
pop    = Population("demo", [Subject("m1", [rec])])

# 2. describe the analysis once — the plan is validated when you build it
ctx = (ContextBuilder()
       .add(GroupOp(by="intensity"))           # 1 -> N: split trials by condition
       .add(WindowOp(pre=0.0, post=0.2))        # align spikes to each event
       .build())

# 3. write the metric for ONE unit; spal runs it across the whole population
def evoked_rate(uc):
    pre, post = uc.cache["window"]
    return uc.cache["csr"].counts.mean() / (post - pre)   # spikes/s in the window

res = apply(pop, evoked_rate, ctx)
# AnalysisResult: one record per (unit × intensity), across the whole cohort

# 4. export for ML
mat = res.to_matrix(rows="unit_id", cols="intensity")     # units × conditions
```

That's the shape of every spal analysis: **plug a source, describe the plan, write
a per-unit metric, export**. You never write the loop over units, recordings, or
conditions — spal walks them and applies your metric to each.

## What each step does

**1 — the source.** `RandomSpikeSource` is a stand-in for your real data. In
practice you write a small [`SpikeSource`](../guides/writing-a-source.md) adapter
for your format, and everything downstream is identical. `Recording.from_source`
reads the source's roster and builds one unit per id.

**2 — the plan.** A `Context` is a lazy plan, not an execution. `GroupOp` splits
each unit's trials by a stimulus condition; `WindowOp` aligns spikes to each event
and produces the per-trial counts. `build()` validates that the ops fit together
(each op's requirements are met) *before* any data is touched — an invalid plan
fails here, not halfway through a long run.

**3 — the metric.** `evoked_rate` takes one `UnitContext` and returns a value.
It's written for a single unit; `apply` runs it across every unit in the
population. The metric reads what the ops produced — here `uc.cache["csr"].counts`
(per-trial spike counts) and `uc.cache["window"]` (the window bounds).

**4 — the export.** `to_matrix` pivots the tidy records into a labelled array. See
[Exporting for ML](../guides/export.md) for `to_dataset`, which gives you an
`(X, y)` design matrix for scikit-learn.

## Inspecting a result

An `AnalysisResult` prints as a one-line summary with a sparkline of the values —
the shape of your result, readable in the terminal without plotting:

```python
>>> res
AnalysisResult(64 records | unit_id×32, intensity×2 | value∈[3.1, 88.4] | GroupOp → WindowOp)
```

You can filter and aggregate before exporting:

```python
res.where(intensity=40)                    # keep one condition
res.aggregate(by="intensity", method="mean")   # mean over units, per condition
res.get_unique_coord_values("intensity")   # {0, 40}
```

## Next steps

- [Writing a SpikeSource](../guides/writing-a-source.md) — connect your own data.
- [Exporting for ML](../guides/export.md) — the two ways to shape a design matrix.
- [API reference](../api/apply.md) — every method in detail.
