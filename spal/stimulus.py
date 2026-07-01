from __future__ import annotations
from typing import Any

import numpy as np
import numpy.typing as npt

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


#TODO(tayheau): do a __repr__
class StimulusTable:
    def __init__(
        self,
        onsets: npt.ArrayLike | None = None,
        **params: np.ndarray
    ) -> None:
        self.onsets = np.asarray(onsets, dtype=float)
        n = len(self.onsets)
 
        validated: dict[str, np.ndarray] = {}
        for name, arr in params.items():
            validated[name] = self._validate_array(arr, n, name)

        order = np.argsort(self.onsets)
        self.onsets = self.onsets[order]
        self.params = ParamNamespace({k: v[order] for k, v in validated.items()})

    def unique_conditions(self, by: str | list[str]) -> list[dict[str, Any]]:
        if isinstance(by, str): by = [by]
        for name in by:
            if name not in self.params:
                raise ValueError(f"Parameter '{name}' not found.\nAvailable: {self.params.keys()}")
        if not by:
            return [{}]
        columns = [self.params.__dict__[k] for k in by]
        seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in zip(*columns):
            if row not in seen:
                seen[row] = {k: v.item() if hasattr(v, "item") else v
                             for k, v in zip(by, row)}
        return list(seen.values())

    def get_unique_conditions(self) -> list[dict[str, Any]]:
        return self.unique_conditions(tuple(self.params.keys()))

    def select_where(self, **conditions: Any) -> StimulusTable:
        for name in conditions:
            if name not in self.params:
                raise ValueError(f"Parameter '{name}' not found.\nAvailable: {self.params.keys()}")
        mask = np.ones(len(self.onsets), dtype=bool)
        for name, value in conditions.items():
            col = self.params.__dict__[name]
            if np.issubdtype(col.dtype, np.floating):
                mask &= np.isclose(col, value)
            else:
                mask &= col == value
        return StimulusTable(
            onsets=self.onsets[mask],
            **{k: v[mask] for k, v in self.params.to_dict().items()}
        )
 
    @staticmethod
    def _validate_array(
        arr: np.ndarray ,
        expected_len: int,
        name: str,
    ) -> np.ndarray :
        arr = np.asarray(arr)
        if len(arr) != expected_len:
            raise ValueError(
                f"{name} length ({len(arr)}) != onsets length ({expected_len})."
            )
        return arr

    def __len__(self) -> int:
        return len(self.onsets)
