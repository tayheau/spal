# import unittest
#
# import numpy as np
# from typing_extensions import override
# from spal.source import RandomSpikeSource
#
# class TestRandomSpikeSource(unittest.TestCase):
#     @override
#     def setUp(self):
#         self.src = RandomSpikeSource(duration=20.0, mean_rate_hz=5.0, n_units=3, seed=0)
#
#     def test_unit_ids(self):
#         self.assertEqual(self.src.unit_ids, ["u0", "u1", "u2"])
#
#     def test_spikes_sorted_and_range(self):
#         s = self.src.spikes("u0")
#         self.assertTrue(np.all(np.diff(s) >= 0))
#         self.assertTrue(np.all((s >= 0) & (s <= 20.0)))
#
#     def test_unkown_unit_raises(self):
#         with self.assertRaises(KeyError):
#             _ = self.src.spikes("gitgud")
#
#     def test_seed_determinism(self):
#         a = RandomSpikeSource(10.0, 5.0, 2, seed=42).spikes("u0")
#         b = RandomSpikeSource(10.0, 5.0, 2, seed=42).spikes("u0")
#         np.testing.assert_array_equal(a, b)
#
# if __name__ == '__main__':
#     _ = unittest.main()
