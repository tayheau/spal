# Exporting for ML

An `AnalysisResult` holds tidy per-unit records. To hand it to scikit-learn you
need a matrix. `to_matrix` and `to_dataset` build that matrix — with the shape
decision made explicit, never the silent reshape that hides a bug.

## Two ways to shape a result

The question every export answers is: **what is one row?** spal gives you two
answers, and they correspond to two different scientific questions.

| method | one row is | `value` per record | use it to |
|---|---|---|---|
| `to_matrix` | a coord (e.g. a unit) | scalar | build a units × conditions table |
| `to_dataset` (coord mode) | a coord (e.g. a unit) | scalar | get `(X, y)` where rows are units |
| `to_dataset` (trial mode) | a single trial | vector | decode a condition from population activity |

The split is really about your *individual*: is it the **neuron** (one summary
value per unit) or the **trial** (the population's response on each stimulus
presentation)?

## `to_matrix` — a labelled table

`to_matrix` pivots records into a 2-D array, with coordinates as row and column
keys. The `value` must be a scalar (one number per cell).

```python
# res: one record per (unit × condition), value = mean firing rate
mat = res.to_matrix(rows="unit_id", cols="modulation")

mat.values   # 2-D array: units × modulation depths
mat.rows     # list of row keys  (unit identifiers)
mat.cols     # list of column keys (modulation values)
```

Rows and columns can be composite. If a unit is only unique across the cohort
when you combine several coords, pass them all:

```python
mat = res.to_matrix(rows=["subject_id", "recording_id", "unit_id"],
                    cols="modulation")
```

Missing cells (a unit with no value for some condition) are filled with `NaN` by
default; pass `fill=` to change it. If two records land in the same cell, pass a
`reduce` callable to combine them (otherwise the last one wins):

```python
mat = res.to_matrix(rows="region", cols="modulation", reduce=np.mean)
```

A common use is a representational similarity matrix — correlate the columns of
the table across the population:

```python
mat = res.to_matrix(rows=["subject_id", "recording_id", "unit_id"],
                    cols=["frequency", "modulation"])
sim = np.corrcoef(mat.values, rowvar=False)   # condition × condition similarity
```

!!! warning "NaN and `np.corrcoef`"
    `np.corrcoef` returns all-NaN if *any* entry is NaN — unlike pandas' pairwise
    correlation. It's only equivalent when every unit has every condition. Check
    `np.isnan(mat.values).any()` first, and drop incomplete units if needed:
    `mat.values[~np.isnan(mat.values).any(axis=1)]`.

## `to_dataset` — a design matrix for scikit-learn

`to_dataset` returns a `Dataset(X, y, features, rows)` — exactly what
`train_test_split` and an estimator's `.fit(X, y)` expect. It has two modes,
chosen by `observation`.

### Coord mode — one row per unit

When `observation` is a coord name, each row is one coordinate tuple (e.g. a
unit), columns are the `features`, and `value` is a scalar. This is the classifier
where you predict something *about a neuron* from its responses across conditions.

```python
ds = res.to_dataset(
    observation="unit_id",       # one row per unit
    features="modulation",       # columns = conditions
    label="region",              # y = what to predict per unit
)

ds.X          # units × conditions, the feature matrix
ds.y          # one label per unit
ds.features   # column keys
ds.rows       # row keys (the units)
```

The label must be **constant within an observation** — a unit can't be in two
regions. If it varies, `to_dataset` raises rather than silently picking one:

```python
# raises ValueError: label 'region' varies within observation ('a',)
# if unit 'a' appears with region "V1" in one record and "M1" in another
```

### Trial mode — one row per trial

When `observation="trial"`, each row is a single trial, columns are the
`features` (units), and the label is the block (condition). This is the population
decoder: predict *which stimulus was played* from the population's activity on
each presentation.

Here `value` must be a **vector** — the per-trial counts for that unit-condition
(e.g. `Csr.counts`), not a scalar summary:

```python
# metric returns a per-trial vector, NOT a mean:
def spikes_per_trial(uc):
    return uc.cache["csr"].counts      # one count per trial

res = apply(pop, spikes_per_trial, ctx)   # ctx groups by condition + windows

ds = res.to_dataset(
    observation="trial",
    features=["subject_id", "recording_id", "unit_id"],  # columns = units
    label="modulation",                                  # y = condition per trial
    align="truncate",
)
# ds.X: (total trials) × (units) ; ds.y: the condition of each trial
```

Then it drops straight into scikit-learn:

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, stratify=ds.y)
clf = SVC(kernel="linear").fit(Xtr, ytr)
print(clf.score(Xte, yte))
```

### `align` is required in trial mode

Units can have different trial counts in the same block — a unit might be silent
for some presentations. spal won't guess how to reconcile that, so `align` is
**required** (there is no default):

- `align="truncate"` cuts every unit to the block's **minimum** trial count. The
  matrix is full, but you drop trials.
- `align="pad"` pads every unit to the block's **maximum** with `fill` (NaN by
  default). You keep all trials, but introduce holes.

```python
ds = res.to_dataset(observation="trial", features="unit_id",
                    label="modulation", align="pad")
```

!!! note "pad and scikit-learn"
    `align="pad"` introduces NaN, and most scikit-learn estimators reject NaN.
    Impute (e.g. `SimpleImputer`) before fitting, or use `truncate` if dropping
    the extra trials is acceptable.

The requirement is deliberate: silently concatenating trials of different lengths
is exactly the shape bug this API exists to prevent. You choose truncate or pad,
in the open.

## Pinning class order across datasets

`label_order` fixes the order of the label classes. This is what makes an analysis
portable across datasets: it pins the class encoding so a classifier sees the same
`y==0` on your data as on someone else's.

```python
ds = res.to_dataset(observation="trial", features="unit_id",
                    label="modulation", align="truncate",
                    label_order=[50, 100, 200, 400])
```

Without it, classes fall back to sorted order — deterministic, but tied to the
values present in *this* dataset. Pinning the order means two runs of the "same"
analysis on different cohorts agree on what each class index means. `label_order`
may list classes absent from a given dataset (they're skipped); present-but-
unlisted classes are appended.

`feature_order` does the same for columns.

## Collisions raise, they don't corrupt

In trial mode, if two records share the same `(block, feature)` — the same unit in
the same condition — spal **raises**, rather than silently overwriting one:

```python
# ValueError: two records share same feature ('a',) in block 'x'.
# trial mode cannot merge them : deduplicate or pool upstream.
```

By contract this can't happen after grouping (one unit per condition), so if you
hit it, your records have a genuine duplicate — dedupe or pool it upstream before
exporting. Using the full identity as features (`["subject_id", "recording_id",
"unit_id"]`) prevents the common case where distinct neurons collide because they
share a local `unit_id`.
