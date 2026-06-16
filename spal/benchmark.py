import os, time
import contextlib

BENCH = int(os.getenv("BENCH", "0"))
_cumu: dict[int, float] = {}
_names: list[str] = []

def reset(): _cumu.clear(); _names.clear()

class Tap:
    def __init__(self, it, i, name):
        self.it, self.i = it, i
        if i == len(_names): _names.append(name)
    def __iter__(self): return self
    def __next__(self):
        t0 = time.perf_counter()
        try: return next(self.it)
        finally: _cumu[self.i] = _cumu.get(self.i, 0.0) + time.perf_counter() - t0

def _report():
    if not _cumu: return
    total = _cumu[max(_cumu)]; w = max(len(n) for n in _names); prev = 0.0
    print("[spal] context")
    for i in sorted(_cumu):
        own = _cumu[i] - prev; prev = _cumu[i]
        print(f"  {_names[i]:<{w}} {own*1e3:8.2f} ms  {own/total*100:4.0f}%")
    print(f"  {'TOTAL':<{w}} {total*1e3:8.2f} ms")

@contextlib.contextmanager
def benchmark():
    global BENCH; old = BENCH; BENCH = 1
    try: yield
    finally: _report(); BENCH = old
