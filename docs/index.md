# spal

**Write metrics, not cohort loops.** Portable neural population analysis from
spike trains.

spal turns a per-unit metric into a cohort-scale analysis. You write one function
that describes what to compute for a single unit; spal runs it across every
subject, recording, and condition, and hands you a tidy result — or a design
matrix for any model. Connect your data once behind a small `SpikeSource`
protocol, and the analysis becomes portable: the same metric runs on anyone's
recordings, and yours run under anyone's analysis.

## Highlights

- Write one [`SpikeSource`](writing_a_source.md), and every analysis runs on it unchanged.
- **Write the metric, not the cohort loop** — function spans every (subject × recording × condition).
- **ML-ready in one call** — `to_matrix` and `to_dataset` hand DL frameworks a feature matrix or `(X, y)`.
- **Rides the ecosystem** — reads NWB, SpikeInterface, your lab baked output: it's the analysis layer, not a new format.

## Usage

```python
from spal.apply import apply

# write the metric for one unit ...
def evoked_rate(uc):
    pre, post = uc.cache["window"]
    return uc.cache["csr"].counts.mean() / (post - pre)

# ... spal runs it across the whole cohort
res = apply(pop, evoked_rate, ctx)
X = res.to_matrix(rows="unit_id", cols="intensity")   # ready for analysis
```

The same `evoked_rate` runs unchanged on one neuron or fifty subjects — the only
thing that changes between datasets is the `SpikeSource` you supply.

See the [Quickstart](quickstart.md) for the full pipeline.

## What spal is

The layer between *sorted units* and *a figure or a feature matrix*. It takes a
spike-sorted recording (from anywhere) and a metric you write once, and produces
tidy records and arrays you can hand to anything.

It is **not** a spike sorter (that's upstream — SpikeInterface, Kilosort), **not**
a data format (that's NWB, which spal consumes), and **not** a plotting library.
It extends the stack instead of asking you to leave it.

## Status

Early and moving. The execution core — hierarchy, context, ops, windowing, tidy
export — is stable in shape; the source adapters and public surface are still
settling. Pin a commit if you build on it.

## Install

```bash
pip install git+https://github.com/tayheau/spal
```

Requires Python ≥ 3.11 and NumPy. `numba` is an optional accelerator for
windowing; `pandas` is only needed for `res.to("pandas")`.
