import unittest 

import numpy as np
from typing_extensions import override

from spal.apply import AnalysisResult, apply
from spal.context import ContextBuilder
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

    def test_values(self):
        self.assertEqual(sorted(self.res.values), [2, 3, 4, 5])

    def test_full_population_no_windowing(self):
        self.assertEqual(self.res.records[0]["value"], 5)  # a0 raw count

    def test_apply_with_context(self):
        ctx = (ContextBuilder()
               .add(StimulusOp())                    # pas de conditions -> tous les onsets
               .add(WindowOp(0.0, 0.5))              # [onset, onset+0.5)
               .build())
        res = apply(self.stim_pop, lambda uc: uc.cache["csr"].counts.sum(), ctx=ctx)
        # u0 fires once per onset (3 onsets) -> 3 ; u1 twice per onset -> 6
        self.assertEqual(sorted(res.values), [3, 6])

if __name__ == "__main__":
    _ = unittest.main()
