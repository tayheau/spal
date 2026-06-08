import unittest 

import numpy as np

from spal.core.hierarchy import Unit, Recording, Population, Subject
from spal.core.apply import apply
from spal.core.context import ContextBuilder, StimulusOp, WindowOp
from helpers import FakeSource, make_population

N_SPIKES = lambda uc: len(uc.spikes)

class TestApply(unittest.TestCase):
    def setUp(self):
        self.pop = make_population()
        self.res = apply(self.pop, N_SPIKES)

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
               .add(StimulusOp(np.array([1.0])))
               .add(WindowOp(-0.5, 0.5))
               .build())
        res = apply(self.pop, lambda uc: len(uc.cache["trials"]), ctx=ctx)
        self.assertTrue(all(v == 1 for v in res.values))  # one event -> one trial

if __name__ == "__main__":
    unittest.main()
