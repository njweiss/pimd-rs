"""Compare two spike result files (E/F lines, eV and eV/Å) or two edge files.

    python spikes/m6-torch/compare.py results A B     -> max |dE|, max |dE|/|E|, max |dF|/max|F|
    python spikes/m6-torch/compare.py edges A B       -> edge sets equal? (sender, receiver, unit
                                                         shift per bead; shifts within 1e-9 Å)
Pure numpy; runs in either pixi environment.
"""

import sys

import numpy as np


def results(path):
    e, f = {}, {}
    for ln in open(path):
        t = ln.split()
        if t[0] == "E":
            e[int(t[1])] = float(t[2])
        elif t[0] == "F":
            f[(int(t[1]), int(t[2]))] = [float(x) for x in t[3:6]]
    keys = sorted(f)
    return np.array([e[b] for b in sorted(e)]), np.array([f[k] for k in keys])


def edges(path):
    out = {}
    for ln in open(path):
        t = ln.split()
        key = (int(t[1]), int(t[2]), int(t[3]), int(t[7]), int(t[8]), int(t[9]))
        out[key] = np.array([float(x) for x in t[4:7]])
    return out


def main():
    mode, a, b = sys.argv[1:4]
    if mode == "results":
        ea, fa = results(a)
        eb, fb = results(b)
        assert ea.shape == eb.shape and fa.shape == fb.shape, (ea.shape, eb.shape, fa.shape, fb.shape)
        de = np.abs(ea - eb)
        df = np.abs(fa - fb).max()
        print(f"beads {len(ea)}  max|dE| {de.max():.3e} eV  max|dE|/|E| {(de / np.abs(eb)).max():.3e}  "
              f"max|dF| {df:.3e} eV/A  max|dF|/max|F| {df / np.abs(fb).max():.3e}  "
              f"(max|F| {np.abs(fb).max():.3f} eV/A, E[0] {eb[0]:.6f} eV)")
    else:
        ea, eb = edges(a), edges(b)
        same = set(ea) == set(eb)
        dmax = max((np.abs(ea[k] - eb[k]).max() for k in ea if k in eb), default=0.0)
        print(f"edges {len(ea)} vs {len(eb)}  same set: {same}  "
              f"only in A: {len(set(ea) - set(eb))}  only in B: {len(set(eb) - set(ea))}  "
              f"max shift diff {dmax:.3e} A")


if __name__ == "__main__":
    main()
