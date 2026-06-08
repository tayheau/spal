from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, TypeVar
from collections.abc import Iterator, Callable
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override

from .hierarchy import Population, Unit

S = TypeVar("S", bound="Op")

def per_unit(
    fn: Callable[[S, UnitContext], UnitContext],
) -> Callable[[S, Iterator[UnitContext]], Iterator[UnitContext]]:
    def __call__(self: S, stream: Iterator[UnitContext]) -> Iterator[UnitContext]:
        return (fn(self, uc) for uc in stream)
    return __call__

@dataclass(slots=True)
class UnitContext:
    """
    Ephemeral runtime object.
    Each operation will enrich the cach with derived data.
    """
    unit: Unit
    coords: dict[str, Any]
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def spikes(self) -> np.ndarray:
        return self.unit.source.spikes(self.unit.id)

class Op(ABC):
    requires: ClassVar[frozenset[str]] = frozenset()
    produces: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def __call__(self, stream: Iterator[UnitContext], /) -> Iterator[UnitContext]:
        ...

@dataclass(frozen=True)
class StimulusOp(Op):
    times: np.ndarray
    produces: ClassVar[frozenset[str]] = frozenset({"events"})

    @override
    @per_unit
    def __call__(self, uc:UnitContext) -> UnitContext:
        cache = dict(uc.cache)
        cache["events"] = self.times
        return replace(uc, cache=cache)

@dataclass(frozen=True)
class WindowOp(Op):
    pre: float
    post: float
    requires: ClassVar[frozenset[str]] = frozenset({"events"})
    produces: ClassVar[frozenset[str]] = frozenset({"window", "trials"})

    @override
    @per_unit
    def __call__(self, uc: UnitContext) -> UnitContext:
        events = uc.cache["events"]
        spikes = uc.spikes
        lo = np.searchsorted(spikes, events + self.pre, side="left")
        hi = np.searchsorted(spikes, events + self.post, side="right")
        trials = [spikes[a:b] - event for event, a, b in zip(events, lo, hi)]
        cache = dict(uc.cache)
        cache["window"] = (self.pre, self.post)
        cache["trials"] = trials
        return replace(uc, cache=cache)

@dataclass(frozen=True)
class Context:
    ops: tuple[Op, ...] = ()

    def stream(self, population: Population) -> Iterator[UnitContext]:
        ucs = (
            UnitContext(unit=unit, coords=coords)
            for coords, unit in population.walk()
        )
        for op in self.ops: ucs = op(ucs)
        yield from ucs

@dataclass(frozen=True)
class ContextBuilder:
    ops: tuple[Op, ...] = ()

    def add(self, op:Op) -> "ContextBuilder":
        return replace(self, ops=self.ops + (op,))

    def build(self) -> Context:
        available: set[str] = set()
        for op in self.ops:
            missing = op.requires - available
            if missing:
                raise ValueError(
                        f'''{type(op).__name__} requires {sorted(missing)},
                        but no prior op produces it.''')
            available |= op.produces
        return Context(ops=self.ops)
