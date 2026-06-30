<div align="center">
<picture>
  <source  srcset="docs/spal.svg">
  <img alt="spal logo" src="docs/spal.svg" width="30%">
</picture>

Portable neural population analysis from spike trains

![CI](https://github.com/tayheau/spal/actions/workflows/ci.yml/badge.svg)
![status: alpha](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.11+-blue)
</div>

---

- Write one `SpikeSource`, and every analysis runs on it unchanged.
- **Write the metric, not the cohort loop** — function spans every (subject × recording × condition).
- **ML-ready in one call** — `to_matrix` and `to_dataset` hand DL frameworks a feature matrix or `(X, y)`.
- **Rides the ecosystem** — reads NWB, SpikeInterface, your lab baked output: it's the analysis layer, not a new format.

## Usage
```python
import numpy as np
from spal.hierarchy import Population, Subject, Recording
from spal.context import ContextBuilder
from spal.ops import GroupOp, WindowOp
from spal.source import RandomSpikeSource
from spal.stimulus import StimulusTable
from spal.apply import apply

# 1. plug your data in behind the SpikeSource protocol (synthetic here)
onsets = np.arange(1.0, 100.0, 2.0)
stim = StimulusTable(onsets, intensity=np.random.choice([0, 40], len(onsets)))
src = RandomSpikeSource(onsets, n_units=32, seed=0)
rec = Recording.from_source("rec0", src, stim)
pop = Population("demo", [Subject("m1", [rec])])

# 2. describe the analysis once — validated at build time
ctx = (ContextBuilder()
      .add(GroupOp(by="intensity"))           # 1 → N: split trials by condition
      .add(WindowOp(pre=-0.05, post=0.20))    # align spikes to each event
      .build())

# 3. write the metric, not the cohort loop
def evoked_rate(uc):
   pre, post = uc.cache["window"]
   return uc.cache["csr"].counts.mean() / (post - pre)   # spikes/s in window

res = apply(pop, evoked_rate, ctx)
# AnalysisResult: one record per (unit × intensity), across the whole population

# 4. ML-ready export
X = res.to_matrix(rows="unit_id", cols="intensity")       # feature matrix
```

The same `evoked_rate` runs unchanged on one neuron or fifty subjects — the only
thing that changes between datasets is the source adapter.

## What spal is not
- **Not a spike sorter** — that's SpikeInterface / Kilosort, upstream of spal.
- **Not a data format** — that's NWB, spal consumes it.
- **Not a plotting library** — it produces tidy records and arrays you can hand to
  anything.

It's the layer between *sorted units* and *a figure or a feature matrix*.

## Installation
### From Source
```bash
git clone https://github.com/tayheau/spal.git
cd spal
python3 -m pip install -e .
```

### From GitHub
```bash
python3 -m pip install git+https://github.com/tayheau/spal
```
Requires Python ≥ 3.11 and numpy.


## Status
Early and moving. The execution core (hierarchy, context, ops, CSR windowing, export) is mostly stable in shape.
### Current focus
- source adapters and public surface
- remove the Numba dependence with custom and fused C kernels
- establish benchmarks for rss
- build the doc
- strenghtening `SpikeSource` protocol with testing and validation tooling to make it easier to use
- write the test suite
- put links in the README claims to make it more explicit and fast to grasp

Pin a commit if you build on it.
