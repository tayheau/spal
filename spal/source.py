from __future__ import annotations

from typing import Literal, Protocol
from pathlib import Path

import numpy as np


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
        duration: float,
        mean_rate_hz: float = 5.0,
        n_units: int = 2,
        seed: int | None = None,
    ):
        self.duration = duration
        self.mean_rate_hz = mean_rate_hz

        self._unit_ids = [f"u{i}" for i in range(n_units)]
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def unit_ids(self) -> list[str]:
        return list(self._unit_ids)

    def spikes(self, unit_id: str) -> np.ndarray:

        if unit_id not in self._unit_ids: raise KeyError(f"Unknown unit '{unit_id}'")

        if unit_id not in self._cache:

            n_spikes = self._rng.poisson(self.duration * self.mean_rate_hz)

            self._cache[unit_id] = np.sort(
                self._rng.uniform(0, self.duration, size=n_spikes)
            )

        return self._cache[unit_id]
