from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from .hierarchy import Population, Unit


@dataclass
class UnitContext:
    """
    Ephemeral runtime object. Carries references + whatever the ops resolved.
    """

    unit: Unit
    spikes: np.ndarray
    coords: dict = field(default_factory=dict)
    events: np.ndarray | None = None
    window: tuple[float, float] | None = None
    trials: list[np.ndarray] | None = None


# --------------------------------------------------------------------------- #
# Ops: each one is a small transformation UnitContext -> UnitContext.
# stream() knows none of them; adding an op never touches stream().
# --------------------------------------------------------------------------- #
@runtime_checkable
class Op(Protocol):
    def apply(self, uc: UnitContext) -> UnitContext: ...


@dataclass(frozen=True)
class StimulusOp:
    """Anchor: attaches the event onsets the window aligns to (seconds)."""

    times: np.ndarray

    def apply(self, uc: UnitContext) -> UnitContext:
        return replace(uc, events=self.times)


@dataclass(frozen=True)
class WindowOp:
    """Cut the train into per-event windows; trials are event-relative (s)."""

    pre: float
    post: float

    def apply(self, uc: UnitContext) -> UnitContext:
        if uc.events is None:
            raise ValueError(
                "WindowOp needs events to anchor to; "
                ".with_stimulus(...) must come before .with_window(...)"
            )
        lo = np.searchsorted(uc.spikes, uc.events + self.pre, side="left")
        hi = np.searchsorted(uc.spikes, uc.events + self.post, side="right")
        trials = [uc.spikes[a:b] - e for e, a, b in zip(uc.events, lo, hi)]
        return replace(uc, window=(self.pre, self.post), trials=trials)


@dataclass(frozen=True)
class Context:
    """
    Immutable execution plan: an ordered tuple of ops. stream() folds them over
    each unit's spikes, lazily, one ephemeral UnitContext at a time.
    """

    ops: tuple[Op, ...] = ()

    def stream(self, population: Population) -> Iterator[UnitContext]:

        for coords, unit in population.walk():

            uc = UnitContext(
                unit=unit,
                spikes=unit.source.spikes(unit.id),
                coords=coords,
            )

            for op in self.ops:
                uc = op.apply(uc)

            yield uc


@dataclass(frozen=True)
class ContextBuilder:
    """
    Immutable DSL. Each step appends an op; ordering carries the semantics.
    """

    ops: tuple[Op, ...] = ()

    def with_stimulus(self, times) -> "ContextBuilder":
        return replace(
            self,
            ops=self.ops + (StimulusOp(np.asarray(times, dtype=float)),),
        )

    def with_window(self, pre: float, post: float) -> "ContextBuilder":
        return replace(
            self,
            ops=self.ops + (WindowOp(float(pre), float(post)),),
        )

    def build(self) -> Context:
        return Context(ops=self.ops)
