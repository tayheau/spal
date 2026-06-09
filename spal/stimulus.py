from __future__ import annotations
from typing import Any

import numpy as np

class ParamNamespace:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)

    def __setattr__(self, name: str, value: Any, /) -> None:
        raise AttributeError(
            '''
            StimulusTable params are read-only.
            Create a new StimulusTable to change params.
            '''
        )    

    def __contains__(self, name: str) -> bool:
        return name in self.__dict__

    def keys(self) -> list[str]:
        """List of available parameter names."""
        return list(self.__dict__.keys())
 
    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a copy as a plain dict."""
        return dict(self.__dict__)


class StimulusTable:
    def __init__(
        self,
        onsets: np.ndarray | None = None,
        **params: np.ndarray
    ) -> None:
        self.onsets = np.asarray(onsets, dtype=float)
        n = len(self.onsets)
 
        validated: dict[str, np.ndarray] = {}
        for name, arr in params.items():
            arr = self._validate_array(arr, n, name)
            validated[name] = arr

        order = np.argsort(self.onsets)
        self.onsets      = self.onsets[order]
        self.params = ParamNamespace({k: v[order] for k, v in validated.items()})

    def get_unique_conditions(self) -> list[dict[str, float]]:
        keys = self.params.keys()
        if not keys:
            return [{}]
        matrix = np.column_stack([self.params.__dict__[k] for k in keys])

        unique_rows = np.unique(matrix, axis = 0)

        return [
            {k: float(row[i]) for i, k in enumerate(keys)}
            for row in unique_rows
        ]

    def select_where(self, **conditions: float) -> StimulusTable:
        for name in conditions:
            if name not in self.params:
                raise ValueError(f"Parameter '{name}' not found.\nAvailable: {self.params.keys()}")

        mask = np.ones(len(self.onsets), dtype=bool)
        for name, value in conditions.items():
            mask &= np.isclose(self.params.__dict__[name], value)

        return StimulusTable(
                onsets = self.onsets[mask],
                **{k: v[mask] for k, v in self.params.to_dict().items()}
            )

 
 
    @staticmethod
    def _validate_array(
        arr: np.ndarray ,
        expected_len: int,
        name: str,
    ) -> np.ndarray :
        arr = np.asarray(arr, dtype=float)
        if len(arr) != expected_len:
            raise ValueError(
                f"{name} length ({len(arr)}) != onsets length ({expected_len})."
            )
        return arr

    def __len__(self) -> int:
        return len(self.onsets)
