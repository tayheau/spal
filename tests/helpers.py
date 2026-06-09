import numpy as np
from spal.hierarchy import Recording, Subject, Population

class FakeSource:
    """Deterministic SpikeSource for exact assertions."""
    def __init__(self, spikes_by_unit: dict[str, list[float]]):
        self._d = {k: np.asarray(v, dtype=float) for k, v in spikes_by_unit.items()}

    @property
    def unit_ids(self) -> list[str]:
        return list(self._d)

    def spikes(self, unit_id: str) -> np.ndarray:
        return self._d[unit_id]


def make_population() -> Population:
    sA = FakeSource({"a0": [0.1, 0.5, 0.9, 1.2, 2.0], "a1": [0.2, 0.3, 1.0]})  # 5, 3
    sB = FakeSource({"b0": [0.4, 0.6], "b1": [1.1, 1.3, 1.5, 1.9]})            # 2, 4
    rA = Recording.from_source("rA", sA, metadata={"region": "V1"})
    rB = Recording.from_source("rB", sB, metadata={"region": "M1"})
    subj = Subject("s0", [rA, rB], metadata={"genotype": "wt"})
    return Population("p0", [subj])
