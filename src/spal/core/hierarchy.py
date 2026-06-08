from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .source import SpikeSource


@dataclass(frozen=True)
class Unit:
    id: str
    source: SpikeSource
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recording:
    id: str
    units: list[Unit]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_source(
        cls,
        rid: str,
        source: SpikeSource,
        *,
        metadata: dict[str, Any] | None = None,
        unit_metadata: dict[str, dict] | None = None,
    ) -> "Recording":
        """Build a Recording from a self-describing source's roster."""
        umd = unit_metadata or {}
        units = [Unit(uid, source, dict(umd.get(uid, {}))) for uid in source.unit_ids]
        return cls(rid, units, metadata or {})


@dataclass(frozen=True)
class Subject:
    id: str
    recordings: list[Recording]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Population:
    id: str
    subjects: list[Subject]
    metadata: dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[tuple[dict, Unit]]:
        """
        Iterate over (coords, unit) for every unit in the population.

        coords carries the ancestry (subject / recording / unit ids) merged
        with metadata, most-specific wins. This is what lets results be grouped
        by subject_id, region, genotype, ... after the fact -- info that is
        irrecoverable once units are flattened.
        """

        for subject in self.subjects:
            for recording in subject.recordings:
                for unit in recording.units:

                    coords = {
                        **subject.metadata,
                        **recording.metadata,
                        **unit.metadata,
                        "subject_id": subject.id,
                        "recording_id": recording.id,
                        "unit_id": unit.id,
                    }

                    yield coords, unit

    def units(self) -> Iterator[Unit]:
        """
        Iterate over every unit (coords dropped).
        """

        for _, unit in self.walk():
            yield unit
