from typing import Any
import unittest

import numpy as np

from spal.hierarchy import Unit, Recording, Population, Subject
from spal.stimulus import StimulusTable
from tests.helpers import FakeSource, make_population, make_unit

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
    def test_no_aliasing_metadata(self):
        r_meta : dict[str, Any] = {"level":"recording"}
        rec = Recording("r", [], metadata=r_meta)
        r_meta.update({"injected":True})
        self.assertNotIn("injected", rec.metadata)

    def test_metadata_modification(self):
        rec = Recording("r", [], metadata = {"level":"recording"})
        rec.metadata.update({"injected":True})
        self.assertTrue(rec.metadata.get("injected"))

    def test_from_source_multiple_unit(self):
        rec = Recording.from_source("r", FakeSource({"a": [1.0], "b":[2.0]}))
        self.assertEqual([u.id for u in rec.units], ["a", "b"])

    def test_from_source_metadata_propagation(self):
        rec = Recording.from_source("r", FakeSource({"a": [1.0], "b":[2.0]}),
                                    metadata={"level":"recording"},
                                    unit_metadata={"a": {"level":"unit", "extra":True}, "b": {"level":"unit"}},
                                    )
        self.assertDictEqual(rec.metadata, {"level":"recording"})
        self.assertDictEqual(rec.units[0].metadata, {"level":"unit", "extra":True})
        self.assertDictEqual(rec.units[1].metadata, {"level":"unit"})

    def test_recording_repr_empty(self):
        rec = Recording("r", [], metadata = {"level":"recording"})
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


class TestView(unittest.TestCase):
    def _rec(self) -> Recording:
        return Recording.from_source("r", FakeSource({"a":[1.0], "b":[2.0]}))

    def test_set_and_get_column(self):
        rec = self._rec()
        rec.unit_metadata["name"] = ["unit_a", "unit_b"]
        self.assertEqual(rec.unit_metadata["name"], ["unit_a", "unit_b"])

    def test_write_through_units(self):
        rec = self._rec()
        rec.unit_metadata["name"] = ["unit_a", "unit_b"]
        self.assertEqual(rec.units[0].metadata.get("name"), "unit_a")
        self.assertEqual(rec.units[1].metadata.get("name"), "unit_b")

    def test_length_mismatch(self):
        rec = self._rec()
        with self.assertRaises(ValueError):
            rec.unit_metadata["name"] = ["unit_a"]
        
    def test_missing_key_is_none_filled_array(self):
        rec = self._rec()
        self.assertEqual(rec.unit_metadata["missing"], [None, None])

    def test_single_key(self):
        rec = self._rec()
        rec.units[0].metadata.update({"name":"unit_a"})
        self.assertEqual(rec.unit_metadata["name"], ["unit_a", None])


class TestSubject(unittest.TestCase):
    def test_no_aliasing_metadata(self):
            s_meta : dict[str, Any] = {"level":"subject"}
            sub = Subject("s", [], metadata=s_meta)
            s_meta.update({"injected":True})
            self.assertNotIn("injected", sub.metadata)


class TestPopulation(unittest.TestCase):
    def test_no_aliasing_metadata(self):
        p_meta: dict[str, Any] = {"level":"population"}
        pop = Population("p", [], metadata=p_meta)
        p_meta.update({"injected":True})
        self.assertNotIn("injected", pop.metadata)

class TestWalk(unittest.TestCase):
    def test_total_unit_walked(self):
        self.assertEqual(len(list(make_population().walk())), 4)

    def test_coords_propagates_all_levels(self):
        coords, _, _ = next(iter(make_population().walk()))
        self.assertEqual(coords["subject_id"], "s0")
        self.assertEqual(coords["recording_id"], "rA")
        self.assertEqual(coords["unit_id"], "a0")
        self.assertEqual(coords["region"], "V1")         # from recording metadata
        self.assertEqual(coords["genotype"], "wt")

    def test_precedence_fine_over_large_grained_metadata(self):
        a = make_unit("a", metadata={"level":"unit"})
        rec = Recording("r", [a], metadata={"level":"recording"})
        sub = Subject("s", [rec], metadata={"level":"subject", "shared":"subject"})
        pop = Population("p", [sub], metadata={"level":"population"})
        coords, _, _ = next(iter(pop.walk()))
        self.assertEqual(coords.get("level"), "unit")
        self.assertEqual(coords.get("shared"), "subject")

    def test_reserved_ids_override_metadata(self):
        a = make_unit("a", metadata={"unit_id": "WRONG"})
        pop = Population("p", [Subject("s", [Recording("r", [a])])])
        coords, _, _ = next(iter(pop.walk()))
        self.assertEqual(coords["unit_id"], "a")

    def test_population_metadata_not_propagated(self):
            # walk does NOT merge population.metadata
            # population is the root, not a coordinate dimension
            pop = Population("p", [Subject("s", [Recording("r", [make_unit("u")])])],
                             metadata={"level": "pop"})
            coords, _, _ = next(iter(pop.walk()))
            self.assertNotIn("level", coords)

    def test_walk_yields_matching_unit_and_recording(self):
        coords, unit, recording = next(iter(make_population().walk()))
        self.assertEqual(unit.id, coords["unit_id"])
        self.assertEqual(recording.id, coords["recording_id"])

if __name__ == "__main__":
    _ = unittest.main()
