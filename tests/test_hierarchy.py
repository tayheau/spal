# import unittest 
#
# import numpy as np
#
# from spal.hierarchy import Unit, Recording, Population, Subject
# from tests.helpers import FakeSource, make_population
#
# class TestHierarchy(unittest.TestCase):
#     def test_unit_spikes_delegates(self):
#         u = Unit("x", FakeSource({"x": [1.0, 2.0]}))
#         np.testing.assert_array_equal(u.spikes, np.array([1.0, 2.0]))
#
#     def test_from_source_roster_and_metadata(self):
#         src = FakeSource({"x": [0.0], "y": [1.0]})
#         rec = Recording.from_source("r", src, unit_metadata={"x": {"depth": 100}})
#         self.assertEqual([u.id for u in rec.units], ["x", "y"])
#         self.assertEqual(rec.units[0].metadata, {"depth": 100})
#         self.assertEqual(rec.units[1].metadata, {})
#
#     def test_walk_count_and_coords(self):
#         coords, _ = next(iter(make_population().walk()))
#         self.assertEqual(coords["subject_id"], "s0")
#         self.assertEqual(coords["recording_id"], "rA")
#         self.assertEqual(coords["unit_id"], "a0")
#         self.assertEqual(coords["region"], "V1")
#         self.assertEqual(coords["genotype"], "wt")
#
#     def test_walk_total(self):
#         self.assertEqual(len(list(make_population().walk())), 4)
#
#     def test_coords_precedence_unit_wins(self):
#         src = FakeSource({"x": [0.0]})
#         rec = Recording.from_source("r", src, metadata={"k": "rec"},
#                                     unit_metadata={"x": {"k": "unit"}})
#         pop = Population("p", [Subject("s", [rec], metadata={"k": "subj"})])
#         coords, _ = next(iter(pop.walk()))
#         self.assertEqual(coords["k"], "unit")  # unit > recording > subject
#
#     def test_iterators(self):
#         pop = make_population()
#         self.assertEqual(len(list(pop.units())), 4)
#         self.assertEqual([r.id for r in pop.recordings()], ["rA", "rB"])
#
# if __name__ == "__main__":
#     _ = unittest.main()
