from __future__ import annotations
from abc import ABC, abstractmethod

from typing import ClassVar, TypeVar, TYPE_CHECKING
from collections.abc import Iterator, Callable
from typing_extensions import override
from dataclasses import dataclass, replace

from spal.stimulus import StimulusTable
from spal.window import window

if TYPE_CHECKING: from .context import UnitContext

S = TypeVar("S", bound="Op")

def per_unit(
    fn: Callable[[S, UnitContext], UnitContext],
) -> Callable[[S, Iterator[UnitContext]], Iterator[UnitContext]]:
    def __call__(self: S, stream: Iterator[UnitContext]) -> Iterator[UnitContext]:
        return (fn(self, uc) for uc in stream)
    return __call__

class Op(ABC):
    requires: ClassVar[frozenset[str]] = frozenset()
    produces: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def __call__(self, stream: Iterator[UnitContext], /) -> Iterator[UnitContext]:
        ...

@dataclass(frozen=True)
class StimulusOp(Op):
    stimulus: StimulusTable
    on: str | None = None
    produces: ClassVar[frozenset[str]] = frozenset({"events"})
 
    @override
    @per_unit
    def __call__(self, uc: UnitContext) -> UnitContext:
        if self.on is None:
            events = self.stimulus.onsets
        else:
            if self.on not in self.stimulus.params:
                raise KeyError(
                    f'''StimulusOp(on={self.on!r}) but the table has no such
                    column. Available: {self.stimulus.params.keys()}'''
                )
            events = self.stimulus.select_where(
                **{self.on: uc.coords[self.on]}
            ).onsets
        cache = dict(uc.cache)
        cache["events"] = events
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
        cache = dict(uc.cache)
        cache["window"] = (self.pre, self.post)
        cache["trials"] = window(uc.spikes, uc.cache["events"], self.pre, self.post)
        return replace(uc, cache=cache)

@dataclass(frozen=True)
class GroupOp(Op):
    stimulus: StimulusTable
    by: tuple[str, ...]
    produces: ClassVar[frozenset[str]] = frozenset({"events"})          

    def __call__(self, stream):              
        for uc in stream:
            for cond in self.stimulus.unique_conditions(self.by):   
                sub = self.stimulus.select_where(**cond)
                cache = dict(uc.cache); cache["events"] = sub.onsets
                yield replace(uc, coords={**uc.coords, **cond}, cache=cache)
