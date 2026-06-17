from typing import TypeAlias

# raster braille
# heatmap

_N:TypeAlias = int|float

def spark(a: list[_N]):
    m = max(a) or 1
    return "".join(" ▁▂▃▄▅▆▇█"[min(8, max(0, int(v / m * 8)))] for v in a)

def hbar(v: _N, w:int = 20, m:float = 1.0):
    f = max(0.0, v) / (m or 1) * w; full = int(f)
    return "█" * w if full >= w else "█" * full + " ▏▎▍▌▋▊▉█"[int((f - full) * 8)]

def rangebar(v: _N, r: tuple[_N, _N], w:int = 20):
    lo, hi = min(r), max(r); i = int((v-lo)/(hi-lo)*(w-1)); _in = i in (0,1)
    l = f"{'─'*(i)}┬{'─'*(w-1- i)}"

    _l = [" "] * w
    def _pl(x: _N, col:int, align:str):
        s = f"{x:g}"
        start = col if align == "l" else col - len(s) + 1 if align == "r" else col - len(s) // 2
        start = max(0, min(w - len(s), start))
        for k, ch in enumerate(s):
            _l[start + k] = ch

    _pl(lo, 0, "l")
    _pl(hi, w - 1, "r")
    if not _in:
        _pl(v, i, "c")
    return l + "\n" + "".join(_l)

def frm(s:str):
    _ls = s.split('\n'); _y = len(_ls) - s.endswith('\n')
    _w = max((len(l) for l in _ls), default=0)
    top = "┌" + "─" * (_w + 2) + "┐"
    bot = "└" + "─" * (_w + 2) + "┘"
    body = [f"│ {l}{' ' * (_w - len(l))} │" for l in _ls]
    return "\n".join([top, *body, bot])

if __name__=="__main__":
    print(spark([0.82956879, 0.862423  , 3.40657084, 8.08008214, 6.52772074,
        2.724846  , 1.3613963 , 4.89527721, 9.59137577, 6.21765914,
        2.33059548, 1.22587269, 3.94045175, 9.60574949, 6.14579055,
        2.33059548, 1.24640657, 3.34086242, 9.35523614, 5.88090349,
        2.11909651, 1.22997947, 3.11088296, 9.02053388, 5.99383984,
        2.10061602, 0.96714579, 0.98767967, 0.86652977, 0.85215606]))
    print(frm(spark([0.82956879, 0.862423  , 3.40657084, 8.08008214, 6.52772074,
        2.724846  , 1.3613963 , 4.89527721, 9.59137577, 6.21765914,
        2.33059548, 1.22587269, 3.94045175, 9.60574949, 6.14579055,
        2.33059548, 1.24640657, 3.34086242, 9.35523614, 5.88090349,
        2.11909651, 1.22997947, 3.11088296, 9.02053388, 5.99383984,
        2.10061602, 0.96714579, 0.98767967, 0.86652977, 0.85215606])))
    print(rangebar(11300, (11000, 11500)))
    print(frm(rangebar(11300, (11000, 11500))))


