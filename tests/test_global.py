from __future__ import annotations

import unittest

import numpy as np

from spal.core.hierarchy import Recording, Subject, Population
from spal.core.context import (
    ContextBuilder, Context, UnitContext, StimulusOp, WindowOp,
)
from spal.core.apply import apply, AnalysisResult


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


class _FixedSource:
    """Minimal SpikeSource: hands back predetermined trains (seconds), so
    windowing can be asserted exactly without Poisson noise."""

    def __init__(self, trains: dict):
        self._trains = {k: np.asarray(v, dtype=float) for k, v in trains.items()}

    @property
    def unit_ids(self):
        return list(self._trains)

    def spikes(self, unit_id):
        return self._trains[unit_id]


def _population() -> Population:
    """Two subjects, known trains, region + genotype metadata. walk order is
    m1/r0/u0, m1/r0/u1, m2/r1/u0."""
    src_a = _FixedSource({
        "u0": [0.05, 0.1, 0.95, 1.05, 1.5, 2.4],
        "u1": [0.5, 1.2, 1.25, 2.1],
    })
    src_b = _FixedSource({"u0": [0.9, 1.0, 1.4, 2.6]})

    rec_a = Recording.from_source(
        "r0", src_a,
        unit_metadata={"u0": {"region": "V1"}, "u1": {"region": "M1"}},
    )
    rec_b = Recording.from_source(
        "r1", src_b, unit_metadata={"u0": {"region": "V1"}},
    )
    return Population("pop", [
        Subject("m1", [rec_a], metadata={"genotype": "wt"}),
        Subject("m2", [rec_b], metadata={"genotype": "ko"}),
    ])


# --------------------------------------------------------------------------- #
# context.py
# --------------------------------------------------------------------------- #
class TestContextBuilder(unittest.TestCase):

    def test_with_window_is_immutable(self):
        cb = ContextBuilder()
        cb2 = cb.with_window(-0.2, 0.5)
        self.assertEqual(cb.ops, ())                 # original untouched
        self.assertEqual(len(cb2.ops), 1)
        self.assertIsInstance(cb2.ops[0], WindowOp)

    def test_with_stimulus_is_immutable(self):
        cb = ContextBuilder()
        cb2 = cb.with_stimulus([1.0, 2.0])
        self.assertEqual(cb.ops, ())
        self.assertIsInstance(cb2.ops[0], StimulusOp)
        np.testing.assert_allclose(cb2.ops[0].times, [1.0, 2.0])

    def test_ops_order_preserved(self):
        ctx = ContextBuilder().with_stimulus([1.0]).with_window(-0.1, 0.2).build()
        self.assertIsInstance(ctx.ops[0], StimulusOp)
        self.assertIsInstance(ctx.ops[1], WindowOp)

    def test_build_carries_ops(self):
        cb = ContextBuilder().with_stimulus([1.0]).with_window(0.0, 0.5)
        ctx = cb.build()
        self.assertIsInstance(ctx, Context)
        self.assertEqual(ctx.ops, cb.ops)


class TestOps(unittest.TestCase):

    def _uc(self, spikes, events=None):
        return UnitContext(
            unit=None,
            spikes=np.asarray(spikes, dtype=float),
            events=None if events is None else np.asarray(events, dtype=float),
        )

    def test_stimulus_op_attaches_events(self):
        uc = self._uc([0.1, 0.2])
        out = StimulusOp(np.array([1.0, 2.0])).apply(uc)
        self.assertIsNone(uc.events)                 # input untouched (replace)
        np.testing.assert_allclose(out.events, [1.0, 2.0])

    def test_window_op_requires_events(self):
        with self.assertRaises(ValueError):
            WindowOp(-0.2, 0.5).apply(self._uc([0.1, 0.2], events=None))

    def test_window_op_aligns_trials(self):
        uc = self._uc([0.05, 0.1, 0.95, 1.05, 1.5, 2.4], events=[1.0, 2.0])
        out = WindowOp(-0.2, 0.5).apply(uc)
        self.assertEqual(len(out.trials), 2)
        np.testing.assert_allclose(out.trials[0], [-0.05, 0.05, 0.5])
        np.testing.assert_allclose(out.trials[1], [0.4])
        self.assertEqual(out.window, (-0.2, 0.5))

    def test_window_op_boundaries(self):
        # spikes exactly at e+pre (left-inclusive) and e+post (right-inclusive)
        out = WindowOp(-0.2, 0.5).apply(self._uc([0.8, 1.5], events=[1.0]))
        np.testing.assert_allclose(out.trials[0], [-0.2, 0.5])


class TestStream(unittest.TestCase):

    def setUp(self):
        self.pop = _population()

    def test_yields_one_per_unit(self):
        self.assertEqual(sum(1 for _ in Context().stream(self.pop)), 3)

    def test_carries_coords(self):
        first = next(Context().stream(self.pop))
        self.assertEqual(first.coords["subject_id"], "m1")
        self.assertEqual(first.coords["recording_id"], "r0")
        self.assertEqual(first.coords["unit_id"], "u0")
        self.assertEqual(first.coords["region"], "V1")
        self.assertEqual(first.coords["genotype"], "wt")  # subject metadata propagated

    def test_empty_context_full_train(self):
        first = next(Context().stream(self.pop))
        self.assertIsNone(first.trials)
        self.assertIsNone(first.events)
        np.testing.assert_allclose(first.spikes, [0.05, 0.1, 0.95, 1.05, 1.5, 2.4])

    def test_windowed_stream(self):
        ctx = ContextBuilder().with_stimulus([1.0, 2.0]).with_window(-0.2, 0.5).build()
        first = next(ctx.stream(self.pop))
        self.assertEqual(len(first.trials), 2)
        np.testing.assert_allclose(first.trials[0], [-0.05, 0.05, 0.5])

    def test_reversed_ops_raise(self):
        ctx = ContextBuilder().with_window(-0.2, 0.5).with_stimulus([1.0]).build()
        with self.assertRaises(ValueError):
            next(ctx.stream(self.pop))


# --------------------------------------------------------------------------- #
# apply.py
# --------------------------------------------------------------------------- #
class TestApply(unittest.TestCase):

    def setUp(self):
        self.pop = _population()
        self.ctx = ContextBuilder().with_stimulus([1.0, 2.0]).with_window(-0.2, 0.5).build()

    def test_one_record_per_unit(self):
        res = apply(self.pop, lambda uc: 1.0, self.ctx)
        self.assertIsInstance(res, AnalysisResult)
        self.assertEqual(len(res), 3)

    def test_record_has_coords_and_value(self):
        res = apply(self.pop, lambda uc: sum(t.size for t in uc.trials), self.ctx)
        rec = res.records[0]
        self.assertEqual(rec["unit_id"], "u0")
        self.assertEqual(rec["region"], "V1")
        self.assertEqual(rec["value"], 4)            # 3 spikes @event1 + 1 @event2

    def test_default_ctx_full_population(self):
        res = apply(self.pop, lambda uc: uc.spikes.size)  # ctx=None
        self.assertEqual(len(res), 3)
        self.assertEqual(res.records[0]["value"], 6)   # u0/m1 full train length

    def test_values_property_in_walk_order(self):
        res = apply(self.pop, lambda uc: uc.coords["unit_id"], self.ctx)
        self.assertEqual(res.values, ["u0", "u1", "u0"])


class TestAnalysisResult(unittest.TestCase):

    def setUp(self):
        self.res = AnalysisResult(
            records=[
                {"subject_id": "m1", "region": "V1", "genotype": "wt", "value": 10.0},
                {"subject_id": "m1", "region": "M1", "genotype": "wt", "value": 20.0},
                {"subject_id": "m2", "region": "V1", "genotype": "ko", "value": 30.0},
                {"subject_id": "m2", "region": "V1", "genotype": "ko", "value": 50.0},
            ],
            context=Context(),
        )

    # -- where ---------------------------------------------------------------
    def test_where_equality(self):
        self.assertEqual(len(self.res.where(region="V1")), 3)

    def test_where_membership(self):
        self.assertEqual(len(self.res.where(region=["V1", "M1"])), 4)

    def test_where_callable(self):
        self.assertEqual(len(self.res.where(value=lambda v: v > 25)), 2)

    def test_where_returns_new(self):
        sub = self.res.where(region="M1")
        self.assertEqual(len(self.res), 4)           # original untouched
        self.assertEqual(len(sub), 1)

    # -- aggregate -----------------------------------------------------------
    def test_aggregate_mean(self):
        agg = self.res.where(region="V1").aggregate(by="subject_id")
        by_subj = {r["subject_id"]: r["value"] for r in agg.records}
        self.assertAlmostEqual(by_subj["m1"], 10.0)
        self.assertAlmostEqual(by_subj["m2"], 40.0)  # (30 + 50) / 2

    def test_aggregate_adds_n(self):
        by_geno = {r["genotype"]: r["n"] for r in self.res.aggregate(by="genotype").records}
        self.assertEqual(by_geno, {"wt": 2, "ko": 2})

    def test_aggregate_preserves_constant_coords(self):
        # genotype is constant within subject -> survives subject aggregation,
        # so a second aggregation by genotype is possible (each subject weighted equally).
        agg = self.res.aggregate(by="subject_id").aggregate(by="genotype")
        self.assertEqual({r["genotype"] for r in agg.records}, {"wt", "ko"})

    def test_aggregate_by_tuple(self):
        agg = self.res.aggregate(by=("subject_id", "region"))
        self.assertEqual(len(agg), 3)                # (m1,V1) (m1,M1) (m2,V1)

    def test_aggregate_stack(self):
        agg = self.res.aggregate(by="genotype", method="stack")
        wt = next(r for r in agg.records if r["genotype"] == "wt")
        np.testing.assert_allclose(np.sort(wt["value"]), [10.0, 20.0])

    def test_aggregate_callable_method(self):
        agg = self.res.aggregate(by="genotype", method=lambda vs: max(vs))
        ko = next(r for r in agg.records if r["genotype"] == "ko")
        self.assertEqual(ko["value"], 50.0)

    def test_aggregate_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            self.res.aggregate(by="genotype", method="bogus")

    # -- to ------------------------------------------------------------------
    def test_to_records_returns_underlying_list(self):
        self.assertIs(self.res.to("records"), self.res.records)

    def test_to_numpy(self):
        np.testing.assert_allclose(np.sort(self.res.to("numpy")), [10.0, 20.0, 30.0, 50.0])

    @unittest.skipUnless(_has_pandas(), "pandas not installed")
    def test_to_pandas(self):
        df = self.res.to("pandas")
        self.assertEqual(len(df), 4)
        self.assertIn("value", df.columns)

    def test_to_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.res.to("bogus")


if __name__ == "__main__":
    unittest.main()
