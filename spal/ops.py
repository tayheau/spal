#TODO(tayheau): add syntactic sugar for ContextBuilder Ops calls
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import ClassVar, TypeVar, TYPE_CHECKING, Any
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
    conditions: tuple[tuple[str, Any], ...] = ()
    produces: ClassVar[frozenset[str]] = frozenset({"events"})
 
    @override
    @per_unit
    def __call__(self, uc: UnitContext) -> UnitContext:
        stim = uc.recording.stimulus
        if stim is None:
            raise ValueError(f"recording {uc.coords['recording_id']!r} doesnt have a StimulusTable")
        events = (stim.onsets if not self.conditions
                  else stim.select_where(**dict(self.conditions)).onsets) 
        cache = dict(uc.cache)
        cache["events"] = events
        return replace(uc, cache=cache)

    @classmethod
    def where(cls, **conditions) -> "StimulusOp":
        return cls(tuple(conditions.items()))


@dataclass(frozen=True)
class WindowOp(Op):
    pre: float
    post: float
    requires: ClassVar[frozenset[str]] = frozenset({"events"})
    produces: ClassVar[frozenset[str]] = frozenset({"window", "csr"})

    @override
    @per_unit
    def __call__(self, uc: UnitContext) -> UnitContext:
        cache = dict(uc.cache)
        cache["window"] = (self.pre, self.post)
        cache["csr"] = window(uc.spikes, uc.cache["events"], self.pre, self.post)
        return replace(uc, cache=cache)

# 1:N
@dataclass(frozen=True)
class GroupOp(Op):
    by: str | list[str]
    produces: ClassVar[frozenset[str]] = frozenset({"events"})          

    @override
    def __call__(self, stream: Iterator[UnitContext]):              
        for uc in stream:
            stim = uc.recording.stimulus
            if stim is None:
                raise ValueError(f"recording {uc.coords['recording_id']!r} doesnt have a StimulusTable")
            for cond in stim.unique_conditions(self.by):   
                sub = stim.select_where(**cond)
                cache = dict(uc.cache); cache["events"] = sub.onsets
                yield replace(uc, coords={**uc.coords, **cond}, cache=cache)
