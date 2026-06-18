from __future__ import annotations

from typing import Protocol, runtime_checkable
from pathlib import Path

import numpy as np


#TODO(tayheau): do a __repr__
@runtime_checkable
class SpikeSource(Protocol):
    """
    Backend abstraction. Any object exposing this API is a valid SpikeSource.
    """

    @property
    def unit_ids(self) -> list[str]:
        """Ids of the units this source can serve."""
        ...

    def spikes(self, unit_id: str) -> np.ndarray:
        """Return spike timestamps in seconds (sorted ascending)."""
        ...


class RandomSpikeSource:
    """
    Synthetic source useful for testing. Backs a fixed roster of `n_units`;
    each unit is an independent homogeneous Poisson process, cached on first use.
    """

    def __init__(
        self,
        onsets: np.ndarray,
        mean_rate_hz: float = 5.0,
        n_units: int = 2,
        seed: int | np.random.Generator | None = None,
    ):
        self._duration: float = float(onsets[-1] + 2)
        self._mean_rate_hz:float = mean_rate_hz
        self._seed = seed
        self._onsets = onsets

        self._unit_ids = [f"u{i}" for i in range(n_units)]
        self._cache: dict[str, np.ndarray] = {}

    @property
    def unit_ids(self) -> list[str]:
        return list(self._unit_ids)

    def spikes(self, unit_id: str) -> np.ndarray:
        if unit_id not in self._unit_ids: raise KeyError(f"Unknown unit '{unit_id}'")

        if unit_id not in self._cache:
            rng = np.random.default_rng(self._seed)
            
            if self._seed is None:
                resp_strength = rng.uniform(0, 5)
            else:
                vals = np.linspace(0, 5, len(self._unit_ids))[::-1]
                _i = int(np.argwhere(np.asarray(self._unit_ids) == unit_id).squeeze())
                resp_strength = vals[_i]

            bg = np.sort(rng.uniform(0, self._duration, rng.poisson(self._mean_rate_hz * self._duration)))
            burst = np.concatenate([
                o + rng.uniform(0.02, 0.06, rng.poisson(resp_strength))        
                for o in self._onsets
            ])
            self._cache[unit_id] = np.sort( np.concatenate([bg, burst]))

        return self._cache[unit_id]


#TODO(tayheau)
class SISortingAnalyzerSpikeSource:
    @classmethod
    def load(cls):
        return None

#TODO(tayheau)
class PhySpikeSource:
    @classmethod
    def load(cls):
        return None
