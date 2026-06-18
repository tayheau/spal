from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Csr:
    aligned: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.aligned[self.offsets[i]:self.offsets[i+1]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)


def _window_numpy(spikes, events, pre, post):
    lo = np.searchsorted(spikes, events + pre,  side="left")
    hi = np.searchsorted(spikes, events + post, side="right")
    offsets = np.empty(len(events) + 1, np.int64)
    offsets[0] = 0
    np.cumsum(hi - lo, out=offsets[1:])               # offsets[i+1] = fin du trial i
    aligned = np.empty(int(offsets[-1]), spikes.dtype)
    for i in range(len(events)):
        aligned[offsets[i]:offsets[i + 1]] = spikes[lo[i]:hi[i]] - events[i]
    return aligned, offsets
    
try:
    from numba import njit

    @njit(cache=True)
    def _window_numba(spikes, events, pre, post):
        n = events.shape[0]
        lo = np.searchsorted(spikes, events + pre,  side="left")
        hi = np.searchsorted(spikes, events + post, side="right")
        offsets = np.empty(n + 1, np.int64)
        offsets[0] = 0
        for i in range(n):
            offsets[i + 1] = offsets[i] + (hi[i] - lo[i])
        aligned = np.empty(offsets[n], spikes.dtype)
        k = 0
        for i in range(n):
            e = events[i]
            for j in range(lo[i], hi[i]):
                aligned[k] = spikes[j] - e
                k += 1
        return aligned, offsets

    _kernel = _window_numba
except ImportError:
    _kernel = _window_numpy

def window(spikes: np.ndarray, events: np.ndarray, pre: float, post: float) -> Csr:
    return Csr(*_kernel(spikes, events, pre, post))
