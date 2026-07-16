import unittest 

import numpy as np
from typing_extensions import override
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from spal.apply import AnalysisResult, apply
from spal.context import ContextBuilder, Context
from spal.hierarchy import Population
from spal.ops import StimulusOp, WindowOp
from tests.helpers import make_population, make_stim_population

N_SPIKES = lambda uc: len(uc.spikes)
class TestApply(unittest.TestCase):
    @override
    def setUp(self):
        self.pop: Population = make_population()
        self.stim_pop: Population = make_stim_population()
        self.res: AnalysisResult = apply(self.pop, N_SPIKES)

    def test_one_record_per_unit(self):
        self.assertEqual(len(self.res), 4)

    def test_record_shape(self):
        r = self.res.records[0]
        self.assertIn("value", r)
        self.assertIn("unit_id", r)
        self.assertIn("region", r)

    def test_full_population_no_windowing(self):
        self.assertEqual(self.res.records[0]["value"], 5)  # a0 raw count

    def test_apply_with_context(self):
        ctx = (ContextBuilder()
               .add(StimulusOp())                    # pas de conditions -> tous les onsets
               .add(WindowOp(0.0, 0.5))              # [onset, onset+0.5)
               .build())
        res = apply(self.stim_pop, lambda uc: uc.cache["csr"].counts.sum(), ctx=ctx)
        # u0 fires once per onset (3 onsets) -> 3 ; u1 twice per onset -> 6
        self.assertEqual(sorted(res.get_values()), [3, 6])

class TestARcore(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult([{"value":1, "mean":1, "std":1}, {"value":2, "mean":2, "std":2}],
                                 Context())
    
    def test_basic_setup(self):
        d = {'records':[{"value":1, "mean":1, "std":1}, {"value":2, "mean":2, "std":2}],
             "context":Context(ops=()), "measures":frozenset({"value"})}
        self.assertDictEqual(self.res.__dict__, d)

class TestARget_values(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult([{"value":1, "mean":1, "std":1, "extra_measure":0},
                                   {"value":2, "mean":2, "std":2, "extra_coord":3}],
                                 Context(), frozenset({"value", "mean", "extra_measure"}))

    def test_none_get_values_measures(self):
        expected = dict.fromkeys(["value", "mean",], [1,2])
        expected.update({"extra_measure":[0, None]})
        self.assertDictEqual(self.res.get_values(), expected)

    def test_get_values_measures_coord(self):
        self.assertDictEqual(self.res.get_values(["mean", "std"]), dict.fromkeys(["mean", "std"], [1,2]))

    def test_get_values_non_common_coord(self):
        self.assertEqual(self.res.get_values("extra_coord"), [None, 3])

    def test_get_values_non_common_measure(self):
        self.assertEqual(self.res.get_values("extra_measure"), [0, None])

    def test_invalid_key_get_values(self):
        with self.assertRaises(KeyError):
            _ = self.res.get_values("non_existant")

class TestARwhere(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult([{"value":1, "mean":1, "std":1}, {"value":2, "mean":2, "std":2}],
                                 Context(), frozenset({"value", "mean",}))

    def test_basic_arg_where(self):
        self.assertEqual(self.res.where(value=1).records, [{"value":1, "mean":1, "std":1}])

    def test_empty_arg_where_return_copy(self):
        self.assertIsInstance(sel:=self.res.where(), AnalysisResult)
        self.assertDictEqual(sel.__dict__, self.res.__dict__)

    def test_invalid_kwarg_where(self):
        with self.assertRaises(KeyError):
           _ = self.res.where(wrong=True)

class TestARaggregate_using(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult(
                [{"unit_id":0, "label":"A", "value":10}, {"unit_id":1, "label":"A", "value":20},
                 {"unit_id":2, "label":"B", "value":30}, {"unit_id":3, "label":"B", "value":40}],
                Context())

    def test_aggregate_by_str(self):
        agg = self.res.aggregate_using("label", lambda c : sum(c["value"]))
        self.assertEqual(len(agg), 2)
        self.assertEqual(sorted(agg.get_values("value")), [30, 70])

    def test_aggregate_by_seq(self):
        agg = self.res.aggregate_using(("label", "unit_id"), lambda c : sum(c["value"]))
        self.assertEqual(len(agg), 4)
        self.assertEqual(sorted(agg.get_values("value")), [10, 20, 30, 40])

    def test_dict_results_set_measures(self):
        agg = self.res.aggregate_using("label", lambda c : {"sum": sum(c["value"]), "n": len(c["value"])})
        self.assertEqual(agg.measures, frozenset({"sum", "n"}))
        self.assertEqual(agg.get_values("label"), ["A", "B"])
    
    def test_constant_coord_kept(self):
        agg = self.res.aggregate_using("label", lambda c: 0)
        self.assertIn("label", agg.coord_keys)

    def test_variable_coord_dropped(self):
        agg = self.res.aggregate_using("label", lambda c: 0)
        self.assertNotIn("unit_id", agg.coord_keys)

    def test_empty_aggregate_all(self):
        agg = self.res.aggregate_using([], lambda c: sum(c["value"]))
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg.get_values("value"), [100])

    def test_aggregate_by_invalid_key(self):
        with self.assertRaises(KeyError):
            _ = self.res.aggregate_using("invalid", lambda c : 0)

    def test_aggregate_by_invalid_key_in_seq(self):
        with self.assertRaises(KeyError):
            _ = self.res.aggregate_using(("unit_id", "invalid"), lambda c : 0)

    def test_aggregate_context_preserved(self):
        agg = self.res.aggregate_using([], lambda c: c)
        self.assertEqual(agg.context, self.res.context)


class TestARaggregate(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult(
                [{"unit_id":0, "label":"A", "value":10}, {"unit_id":1, "label":"A", "value":20},
                 {"unit_id":2, "label":"B", "value":30}, {"unit_id":3, "label":"B", "value":40}],
                Context())

    @staticmethod
    def _measure_per_label(agg):
        return {r["label"]:r["value"] for r in agg.records}

    def test_aggregate_sum(self):
        agg = self.res.aggregate("label", "sum")
        self.assertDictEqual(self._measure_per_label(agg), {"A":30, "B":70})

    def test_aggregate_mean(self):
        agg = self.res.aggregate("label", "mean")
        self.assertDictEqual(self._measure_per_label(agg), {"A":15, "B":35})

    def test_aggregate_std(self):
        agg = self.res.aggregate("label", "std")
        self.assertDictEqual(self._measure_per_label(agg), {"A":5, "B":5})

    def test_aggregate_median(self):
        agg = self.res.aggregate("label", "median")
        self.assertDictEqual(self._measure_per_label(agg), {"A":15, "B":35})

    def test_aggregate_stack(self):
        agg = self.res.aggregate("label", "stack")
        self.assertDictEqual(self._measure_per_label(agg), {"A":[10, 20], "B":[30, 40]})

    def test_aggregate_none_collapes(self):
        agg = self.res.aggregate()
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg.get_values(), [25])

    def test_aggregate_selected_measure(self):
        res = AnalysisResult(
            [{"region": "A", "x": 1, "y": 10},
             {"region": "A", "x": 3, "y": 20}],
            Context(), frozenset({"x", "y"}))
        agg = res.aggregate("region", "sum", measure=["x"])
        self.assertEqual(agg.records[0]["x"], 4)
        self.assertNotIn("y", agg.records[0])
        self.assertEqual(agg.measures, frozenset({"x"}))

class TestARto(unittest.TestCase):
    def setUp(self):
        self.res = AnalysisResult(
                [{"unit_id":0, "label":"A", "value":10}, {"unit_id":1, "label":"A", "value":20},
                 {"unit_id":2, "label":"B", "value":30}, {"unit_id":3, "label":"B", "value":40}],
                Context())
        self.res_extra = AnalysisResult(
                [{"unit_id":0, "label":"A", "value":10}, {"unit_id":1, "label":"A", "value":20},
                 {"unit_id":2, "label":"B", "value":30, "extra":777}, {"unit_id":3, "label":"B", "value":40}],
                Context())

    def test_wrong_format_to(self):
        with self.assertRaises(ValueError):
            _ = self.res.export("wrong")

    def test_pandas_format_to(self):
        df = self.res.export('pandas')
        df1 = DataFrame({"unit_id":[0,1,2,3], "label":["A", "A", "B", "B"], "value":[10,20,30,40]})
        assert_frame_equal(df, df1)

    def test_pandas_format_to_extra(self):
        df = self.res_extra.export('pandas')
        df1 = DataFrame({"unit_id":[0,1,2,3], "label":["A", "A", "B", "B"], "value":[10,20,30,40], "extra":[None, None, 777, None]})
        assert_frame_equal(df, df1)

    def test_numpy_format_to(self):
        records = self.res.export("numpy")
        rec = records.view(np.recarray)
        np.testing.assert_array_equal(rec.unit_id, [0,1,2,3])
        np.testing.assert_array_equal(rec.label, ["A", "A", "B", "B"])
        np.testing.assert_array_equal(rec.value, [10,20,30,40])
        dtypes = np.dtype([("unit_id", "<i8"), ("label", "<U1"), ("value", "<i8")])
        np.testing.assert_array_equal(records.dtype, dtypes)

    def test_numpy_format_to_extra(self):
        records = self.res_extra.export("numpy")
        rec = records.view(np.recarray)
        np.testing.assert_array_equal(rec.unit_id, [0,1,2,3])
        np.testing.assert_array_equal(rec.label, ["A", "A", "B", "B"])
        np.testing.assert_array_equal(rec.value, [10,20,30,40])
        np.testing.assert_array_equal(rec.extra, [np.nan,np.nan, 777, np.nan])
        dtypes = np.dtype([("unit_id", "<i8"), ("label", "<U1"),  ("value", "<i8"), ("extra", "<f8"), ])
        np.testing.assert_array_equal(records.dtype, dtypes)

if __name__ == "__main__":
    _ = unittest.main()
