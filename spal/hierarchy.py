from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterator
from typing_extensions import override

from spal.stimulus import StimulusTable
from spal.source import SpikeSource


@dataclass(frozen=True)
class Unit:
    id: str
    source: SpikeSource
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spikes(self):
        return self.source.spikes(self.id)


@dataclass(frozen=True)
class Recording:
    id: str
    units: list[Unit]
    stimulus: StimulusTable | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_source(
        cls,
        id: str,
        source: SpikeSource,
        stimulus: StimulusTable | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        unit_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> "Recording":
        """Build a Recording from a self-describing source's roster."""
        umd = unit_metadata or {}
        units = [Unit(uid, source, umd.get(uid, {})) for uid in source.unit_ids]
        return cls(id, units, stimulus, metadata or {})


    def set_column(self, key: str, values) -> None:
        if len(values) != len(self.units):
            raise ValueError(f"expected {len(self.units)}, got {key}:{len(values)}")
        for _u, _v in zip(self.units, values):
            _u.metadata[key] = _v.item() if hasattr(_v, "item") else _v

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.id!r} | {len(self.units)} units | {len(self.stimulus.onsets)} stimulus"

@dataclass(frozen=True)
class Subject:
    id: str
    recordings: list[Recording]
    metadata: dict[str, Any] = field(default_factory=dict)

    @override
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} {self.id!r} | {len(self.recordings)} recordings"
                +"".join("\n └" + repr(rec) for rec in self.recordings))

@dataclass(frozen=True)
class Population:
    id: str
    subjects: list[Subject]
    metadata: dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[tuple[dict, Unit, Recording]]:
        for subject in self.subjects:
            for recording in subject.recordings:
                for unit in recording.units:
                    coords = { **subject.metadata, **recording.metadata, **unit.metadata,
                        "subject_id": subject.id, "recording_id": recording.id,
                        "unit_id": unit.id,
                    }

                    yield coords, unit, recording

    def units(self) -> Iterator[Unit]:
        """
        Iterate over every unit (coords dropped).
        """

        for _, unit, _ in self.walk():
            yield unit

    def recordings(self) -> Iterator[Recording]:
        """
        Iterate over each recordings.
        """
        for subject in self.subjects:
            yield from subject.recordings
