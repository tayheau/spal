import unittest 

import numpy as np
from typing_extensions import override

from spal.apply import AnalysisResult, apply
from spal.context import ContextBuilder, Context
from spal.hierarchy import Population
from spal.ops import StimulusOp, WindowOp
from tests.helpers import make_population, make_stim_population

N_SPIKES = lambda uc: len(uc.spikes)
#TODO(tayheau): real test suite 
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


if __name__ == "__main__":
    _ = unittest.main()
