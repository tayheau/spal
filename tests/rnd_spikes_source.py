import unittest
import numpy as np 

from spal.core.source import RandomSpikeSource

DURATION = 20.0
HZ = 15
N = 5
SEED = 0

class TestSpikeSource(unittest.TestCase):
    def test_rnd_source(self):
        rnd_source = RandomSpikeSource(DURATION, HZ, N, SEED)
        self.assertEqual(len(rnd_source.unit_ids), N)
        self.assertIsNotNone(rnd_source.spikes("u0"))

if __name__ == '__main__':
    unittest.main()
