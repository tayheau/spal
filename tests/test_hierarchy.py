from typing import Any
import unittest

import numpy as np

from spal.hierarchy import Unit, Recording, Population, Subject
from spal.stimulus import StimulusTable
from tests.helpers import FakeSource

def make_unit(id:str, size:int = 1, metadata: dict[str, Any] | None = None) -> Unit:
    assert (size > 0)
    src = FakeSource({id: np.linspace(0, size, size)})
    return Unit(id, src, metadata=metadata)

class TestUnit(unittest.TestCase):
    def test_no_aliasing_unit_metadata(self):
        metadata: dict[str, Any] = {"level": "unit"}
        a = make_unit("a", metadata=metadata)
        metadata.update({"injected":True})
        self.assertNotIn("injected", a.metadata)

    def test_unit_metadata_modification(self):
        metadata = {"level": "unit"}
        a = make_unit("a", metadata=metadata)
        a.metadata.update({"injected":True})
        self.assertTrue(a.metadata.get("injected"))


class TestRecording(unittest.TestCase):
    def test_no_aliasing_recording_metadata(self):
        r_meta : dict[str, Any] = {"level":"recording"}
        rec = Recording("r", [], recording_metadata=r_meta)
        r_meta.update({"injected":True})
        self.assertNotIn("injected", rec.recording_metadata)

    def test_recording_metadata_modification(self):
        rec = Recording("r", [], recording_metadata = {"level":"recording"})
        rec.recording_metadata.update({"injected":True})
        self.assertTrue(rec.recording_metadata.get("injected"))

    def test_from_source_multiple_unit(self):
        rec = Recording.from_source("r", FakeSource({"a": [1.0], "b":[2.0]}))
        self.assertEqual([u.id for u in rec.units], ["a", "b"])

    def test_from_source_metadata_propagation(self):
        rec = Recording.from_source("r", FakeSource({"a": [1.0], "b":[2.0]}),
                                    metadata={"level":"recording"},
                                    unit_metadata={"a": {"level":"unit", "extra":True}, "b": {"level":"unit"}},
                                    )
        self.assertDictEqual(rec.recording_metadata, {"level":"recording"})
        self.assertDictEqual(rec.units[0].metadata, {"level":"unit", "extra":True})
        self.assertDictEqual(rec.units[1].metadata, {"level":"unit"})

    def test_recording_repr_empty(self):
        rec = Recording("r", [], recording_metadata = {"level":"recording"})
        _repr = repr(rec)
        self.assertIn("Recording 'r'", _repr)
        self.assertIn("0 units", _repr)
        self.assertIn("0 stimulus", _repr)

    def test_recording_repr(self):
        stims = StimulusTable(onsets=[0.5, 1.2, 1.75])
        rec = Recording.from_source("r", FakeSource({"a": [1.0], "b":[2.0]}), stims)
        _repr = repr(rec)
        self.assertIn("Recording 'r'", _repr)
        self.assertIn("2 units", _repr)
        self.assertIn("3 stimulus", _repr)

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
if __name__ == "__main__":
    _ = unittest.main()
