from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Iterator

import numpy as np

from .hierarchy import Population, Unit
from .ops import Op
from . import benchmark

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

@dataclass(frozen=True)
class Context:
    ops: tuple[Op, ...] = ()

    def stream(self, population: Population) -> Iterator[UnitContext]:
        ucs = (
            UnitContext(unit=unit, coords=coords)
            for coords, unit in population.walk()
        )
        if not benchmark.BENCH:
            for op in self.ops: ucs = op(ucs)
        else:
            benchmark.reset()
            for i, op in enumerate(self.ops):
                ucs = benchmark.Tap(op(ucs), i, type(op).__name__)
        yield from ucs

# TODO (tayheau): make a base class with a method for each op 
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
