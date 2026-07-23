#TODO(tayheau): add syntactic sugar for ContextBuilder Ops calls
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import ClassVar, TypeVar, TYPE_CHECKING, Any
from collections.abc import Iterator, Callable
from typing_extensions import override
from dataclasses import dataclass, replace

# from spal.stimulus import StimulusTable
from spal.window import window

if TYPE_CHECKING: from .context import UnitContext, Recording

S = TypeVar("S", bound="Op")
Stream = Iterator["UnitContext"]

def per_unit(
    fn: Callable[[S, UnitContext], UnitContext],
) -> Callable[[S, Stream], Stream]:
    def __call__(self: S, stream: Stream, /) -> Iterator[UnitContext]:
        return (fn(self, uc) for uc in stream)
    return __call__

def per_recording(
    fn: Callable[[S, Recording], dict[str, Any]],
) -> Callable[[S, Stream], Stream]:
    def __call__(self: S, stream: Stream) -> Stream:
        last: Recording | None = None
        delta: dict[str, Any] = {}
        for uc in stream:
            if uc.recording is not last:
                last, delta = uc.recording, fn(self, uc.recording)
            cache = dict(uc.cache); cache.update(delta)
            yield replace(uc, cache=cache)
    return __call__

class Op(ABC):
    requires: ClassVar[frozenset[str]] = frozenset()
    produces: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def __call__(self, stream: Stream, /) -> Stream:
        ...

@dataclass(frozen=True)
class StimulusOp(Op):
    conditions: tuple[tuple[str, Any], ...] = ()
    produces: ClassVar[frozenset[str]] = frozenset({"events"})
 
    @override
    @per_recording
    def __call__(self, rec: Recording) -> dict[str, Any]:
        stim = rec.stimulus
        if stim is None:
            raise ValueError(f"recording {rec.id!r} doesnt have a StimulusTable")
        events = (stim if not self.conditions
                  else stim.select_where(**dict(self.conditions))) 
        return {"events": events}

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
        cache["csr"] = window(uc.spikes, uc.cache["events"].onsets, self.pre, self.post)
        return replace(uc, cache=cache)

@dataclass(frozen=True)
class WhereOp(Op):
    """
    # equal  | WhereOp.where(reactive=True)                    
    # inside  | WhereOp.where(region=("CN", "IC"))              
    # logical AND  | WhereOp.where(genotype="zebra", reactive=True)  
    # function  | WhereOp.where(z=lambda x: x is not None and x > 2)  
    """
    conditions: tuple[tuple[str, Any], ...] = ()
    requires: ClassVar[frozenset[str]] = frozenset()
    produces: ClassVar[frozenset[str]] = frozenset()

    @staticmethod
    def _match(coords: dict[str, Any], conditions: dict[str, Any]) -> bool:
        for k, v in conditions.items():
            g = coords.get(k)
            if callable(v):
                if not v(g):
                    return False
            elif isinstance(v, (list, tuple, set)):
                if g not in v:
                    return False
            elif g != v:
                return False
        return True

    @override
    def __call__(self, stream: Stream) -> Stream:
        conds = dict(self.conditions)
        for uc in stream:
            if self._match(uc.coords, conds):
                yield uc

    @classmethod
    def where(cls, **conditions) -> "WhereOp":
        return cls(tuple(conditions.items()))

# 1:N
@dataclass(frozen=True)
class GroupOp(Op):
    by: str | tuple[str, ...]
    produces: ClassVar[frozenset[str]] = frozenset({"events"})

    @override
    def __call__(self, stream: Stream):
        last_rec = None
        groups: dict[int, list] = {}
        for uc in stream:
            if uc.recording is not last_rec:                   
                groups.clear(); last_rec = uc.recording        
            tab = uc.cache.get("events") or uc.recording.stimulus
            if tab is None:
                raise ValueError(f"recording {uc.coords['recording_id']!r} has no StimulusTable")
            g = groups.get(id(tab))
            if g is None:                                      
                g = groups[id(tab)] = [
                    (cond, tab.select_where(**cond)) for cond in tab.unique_conditions(self.by)
                ]
            for cond, sub in g:
                cache = dict(uc.cache); cache["events"] = sub
                yield replace(uc, coords={**uc.coords, **cond}, cache=cache)
