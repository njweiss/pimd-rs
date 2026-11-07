"""Run-health checks of the M8b plan ("What counts as agreement", run health) for one run directory.

    python3 studies/m8-nma-water/health.py studies/m8-nma-water/out/<set>/<run> [...]

Per run: the largest NMA bond length over every bead of every trajectory frame relative to the
start structure's (limit 1.3), the largest distance of any H from its own heavy atom (limit 1.3 Å;
a larger one is a proton transfer or a dissociation), and the mean T_est over production. Bonds
are taken from the start structure (out/start/nma_w216.xyz): NMA pairs closer than 1.6 Å, each
water's O with the two H that follow it. Minimum image in the cubic box (C-2). Joins part<k>/
outputs of restarted runs only as far as reading frames goes (every frame is checked).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_NMA = 12


def read_xyz_frames(path):
    with open(path) as f:
        while True:
            head = f.readline()
            if not head:
                return
            n = int(head)
            comment = f.readline()
            rows = [f.readline().split() for _ in range(n)]
            yield comment.strip(), [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def box_length(run):
    for line in (run / "config.toml").read_text().splitlines():
        if line.startswith("box_angstrom"):
            return float(line.split("=")[1])
    raise ValueError("no box_angstrom")


def mic(d, L):
    return d - L * np.round(d / L)


def bonds(symbols, x, L):
    pairs = []
    for i in range(N_NMA):
        for j in range(i + 1, N_NMA):
            if np.linalg.norm(mic(x[j] - x[i], L)) < 1.6:
                pairs.append((i, j))
    nma = np.array(pairs)
    hpairs = [(i, j) for i, j in pairs if "H" in (symbols[i], symbols[j])]
    hpairs = [(i, j) if symbols[j] == "H" else (j, i) for i, j in hpairs]
    for o in range(N_NMA, len(symbols), 3):
        assert symbols[o] == "O" and symbols[o + 1] == symbols[o + 2] == "H", o
        hpairs += [(o, o + 1), (o, o + 2)]
    return nma, np.array(hpairs)


def check(run):
    L = box_length(run)
    _, sym0, x0 = next(read_xyz_frames(HERE / "out/start/nma_w216.xyz"))
    nma, xh = bonds(sym0, x0, L)
    r0 = np.linalg.norm(mic(x0[nma[:, 1]] - x0[nma[:, 0]], L), axis=1)
    trajs = sorted(run.glob("part*/trajectory.xyz")) + [run / "trajectory.xyz"]
    worst_ratio, worst_xh, frames = 0.0, 0.0, 0
    for t in trajs:
        for comment, _, x in read_xyz_frames(t):
            n = int(comment.split("atoms=")[1].split()[0])
            for b in x.reshape(-1, n, 3):
                r = np.linalg.norm(mic(b[nma[:, 1]] - b[nma[:, 0]], L), axis=1)
                worst_ratio = max(worst_ratio, float((r / r0).max()))
                worst_xh = max(worst_xh, float(np.linalg.norm(mic(b[xh[:, 1]] - b[xh[:, 0]], L), axis=1).max()))
            frames += 1
    obs = np.genfromtxt(run / "observables.csv", delimiter=",", names=True)
    ok = worst_ratio < 1.3 and worst_xh < 1.3
    print(f"{run}: {frames} frames, max NMA bond / start {worst_ratio:.3f}, max X-H {worst_xh:.3f} Å, "
          f"mean T_est {obs['t_est_k'].mean():.2f} K over {len(obs)} rows: {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([check(Path(a)) for a in sys.argv[1:]]) else 1)
