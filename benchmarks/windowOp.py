import numpy as np, timeit
from numba import njit
import os, subprocess, ctypes, timeit


_C_SRC = r'''
void fill(const double* spikes, const double* events,
          const long* lo, const long* counts, const long* offsets,
          long n, double* out) {
    for (long i = 0; i < n; ++i) {
        double e = events[i];
        const double* src = spikes + lo[i];
        double* dst = out + offsets[i];
        for (long j = 0, c = counts[i]; j < c; ++j) dst[j] = src[j] - e;
    }
}
'''
_SO = "/tmp/_csr_kern.so"
if not os.path.exists(_SO) or os.getenv("REBUILD"):
    with open("/tmp/_csr_kern.c", "w") as f:
        f.write(_C_SRC)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
         "-o", _SO, "/tmp/_csr_kern.c"], check=True)

_lib = ctypes.CDLL(_SO)
_dp = ctypes.POINTER(ctypes.c_double)
_lp = ctypes.POINTER(ctypes.c_long)
_lib.fill.argtypes = [_dp, _dp, _lp, _lp, _lp, ctypes.c_long, _dp]
def _d(a): return a.ctypes.data_as(_dp)
def _l(a): return a.ctypes.data_as(_lp)

def window_list(spikes, events, pre, post):
    lo = np.searchsorted(spikes, events + pre,  side="left")
    hi = np.searchsorted(spikes, events + post, side="right")
    offsets = np.empty(len(events) + 1, np.int64)
    offsets[0] = 0
    np.cumsum(hi - lo, out=offsets[1:])               # offsets[i+1] = fin du trial i
    aligned = np.empty(int(offsets[-1]), spikes.dtype)
    for i in range(len(events)):
        aligned[offsets[i]:offsets[i + 1]] = spikes[lo[i]:hi[i]] - events[i]
    return aligned, offsets

def window_csr(spikes, events, pre, post):
    lo = np.searchsorted(spikes, events + pre, side="left")
    hi = np.searchsorted(spikes, events + post, side="right")
    counts = hi - lo
    offsets = np.empty(len(events) + 1, np.int64); offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    ramp = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
    src = np.repeat(lo, counts) + ramp
    aligned = spikes[src] - np.repeat(events, counts)
    return aligned, offsets

@njit(cache=True)
def numba_csr(spikes, events, pre, post):
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

def csr_c(spikes, events, pre, post):                # fused C kernel, single pass
    lo = np.searchsorted(spikes, events + pre, "left")
    hi = np.searchsorted(spikes, events + post, "right")
    counts = hi - lo
    offsets = np.empty(len(events) + 1, np.int64); offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    out = np.empty(int(offsets[-1]), np.float64)
    _lib.fill(_d(spikes), _d(events), _l(lo), _l(counts), _l(offsets),
              len(events), _d(out))
    return out, offsets

def measure(f, args, repeat=7):
    """min over `repeat` runs, autorange picks the inner number. ms."""
    t = timeit.Timer(lambda: f(*args))   # timeit disables cyclic gc by default
    number, _ = t.autorange()
    return min(t.repeat(repeat=repeat, number=number)) / number * 1e3

rng = np.random.default_rng(0)
DUR, PRE, POST = 3600.0, 0.0, 0.12

print(f"{'n_spikes':>9} {'n_events':>9} {'spk/trial':>9} {'list ms':>9} {'csr ms':>9} {'numba ms':>9} {'c ms':>9} {'speedup csr':>8} {'speedup numba':>8} {'speedup c':>8}")
print("-" * 80)
for n_spk in (500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000):
    spikes = np.sort(rng.uniform(0, DUR, n_spk))
    for n_ev in (60, 500, 5_000, 50_000):
        events = np.sort(rng.uniform(0, DUR - 1, n_ev))
        spt = (POST - PRE) * n_spk / DUR
        t0 = measure(window_list, (spikes, events, PRE, POST))
        t1 = measure(window_csr,  (spikes, events, PRE, POST))
        t2 = measure(numba_csr,  (spikes, events, PRE, POST))
        t3 = measure(csr_c,  (spikes, events, PRE, POST))

        print(f"{n_spk:>9} {n_ev:>9} {spt:>9.1f} {t0:>9.3f} {t1:>9.3f} {t2:>9.3f} {t3:>9.3f} {t0/t1:>7.2f}x {t0/t2:>7.2f}x {t0/t3:>7.2f}x")

#  n_spikes  n_events spk/trial   list ms    csr ms  numba ms      c ms speedup csr speedup numba speedup c
# --------------------------------------------------------------------------------
#    500000        60      16.7     0.026     0.009     0.003     0.011    2.97x    9.08x    2.43x
#    500000       500      16.7     0.239     0.057     0.022     0.033    4.22x   10.91x    7.33x
#    500000      5000      16.7     2.493     0.772     0.514     0.533    3.23x    4.85x    4.67x
#    500000     50000      16.7    22.164     6.247     3.434     3.371    3.55x    6.45x    6.57x
#   1000000        60      33.3     0.025     0.011     0.003     0.011    2.25x    7.28x    2.27x
#   1000000       500      33.3     0.238     0.092     0.025     0.049    2.60x    9.41x    4.84x
#   1000000      5000      33.3     2.657     1.102     0.691     0.695    2.41x    3.85x    3.82x
#   1000000     50000      33.3    24.102    11.652     4.264     4.008    2.07x    5.65x    6.01x
#   1500000        60      50.0     0.025     0.013     0.004     0.011    1.96x    6.39x    2.21x
#   1500000       500      50.0     0.288     0.137     0.039     0.080    2.10x    7.44x    3.60x
#   1500000      5000      50.0     2.787     1.492     0.823     0.817    1.87x    3.39x    3.41x
#   1500000     50000      50.0    27.123    25.690     5.137     4.716    1.06x    5.28x    5.75x
#   2000000        60      66.7     0.026     0.015     0.004     0.012    1.72x    6.03x    2.23x
#   2000000       500      66.7     0.302     0.161     0.040     0.081    1.87x    7.60x    3.71x
#   2000000      5000      66.7     2.942     1.808     0.935     0.917    1.63x    3.15x    3.21x
#   2000000     50000      66.7    29.465    30.734     5.970     5.463    0.96x    4.94x    5.39x
#   2500000        60      83.3     0.026     0.016     0.005     0.012    1.64x    5.56x    2.16x
#   2500000       500      83.3     0.306     0.193     0.071     0.094    1.59x    4.31x    3.24x
#   2500000      5000      83.3     3.035     2.184     1.043     1.011    1.39x    2.91x    3.00x
#   2500000     50000      83.3    31.293    38.294     7.003     6.232    0.82x    4.47x    5.02x
