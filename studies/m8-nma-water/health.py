"""Run-health checks of the M8b plan ("What counts as agreement", run health, as amended
2026-11-08) for one run directory.

    python3 studies/m8-nma-water/health.py studies/m8-nma-water/out/<set>/<run> [...]

Per run, in the bead centroids of every trajectory frame: the largest NMA bond length relative to
the start structure's (limit 1.3) and the largest distance of any H from its own heavy atom
(limit 1.3 Å), and whether any H centroid is closer to another heavy atom than to its own (a
proton transfer). The same two distances over every bead are reported without a limit. Then
|T_est − 300 K − b_P| < 5σ + 0.5 K, with b_P the BAOAB kinetic bias of tbias_harmonic.py and σ
the largest block standard error over 1, 2 and 4 ps blocks, and
|⟨E_prim⟩ − ⟨E_vir⟩ − d_P| < 5σ_c + 0.25 |d_P|, with d_P the step-size bias of ebias_harmonic.py.
Bonds are taken from the start structure (out/start/nma_w216.xyz): NMA pairs closer than 1.6 Å,
each water's O with the two H that follow it. Minimum image in the cubic box (C-2). Restarted
runs: part<k>/ outputs are joined, each cut before the first step of the next part.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_NMA = 12
BIAS_K = {1: -2.29, 8: -2.30, 16: -2.33, 32: -2.46}  # tbias_harmonic.py, Δt = 0.5 fs
BIAS_D_EH = {1: 0.0, 8: -8.80e-6, 16: -3.22e-5, 32: -8.95e-5}  # ebias_harmonic.py, Δt = 0.5 fs
ROWS_PER_PS = 2000  # Δt = 0.5 fs, stride 1


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


def parts(run, name):
    return sorted(run.glob(f"part*/{name}"), key=lambda p: int(p.parent.name[4:])) + [run / name]


def frames(run):
    """(step, beads [P, N, 3]) of every trajectory frame, parts joined."""
    files = parts(run, "trajectory.xyz")
    firsts = [int(next(read_xyz_frames(f))[0].split("step=")[1].split()[0]) for f in files]
    for f, stop in zip(files, firsts[1:] + [np.inf]):
        for comment, _, x in read_xyz_frames(f):
            step = int(comment.split("step=")[1].split()[0])
            if step >= stop:
                break
            n = int(comment.split("atoms=")[1].split()[0])
            yield step, x.reshape(-1, n, 3)


def observables(run):
    # A part cut by a crash can end in a partial line; it lies past the checkpoint and is dropped.
    tables = [np.genfromtxt(f, delimiter=",", names=True, invalid_raise=f.parent == run)
              for f in parts(run, "observables.csv")]
    keep = [t[t["step"] < nxt["step"][0]] for t, nxt in zip(tables, tables[1:])] + [tables[-1]]
    return np.concatenate(keep)


def block_sem(x):
    s = []
    for ps in (1, 2, 4):
        b = ps * ROWS_PER_PS
        m = len(x) // b
        s.append(x[: m * b].reshape(m, b).mean(1).std(ddof=1) / np.sqrt(m))
    return max(s)


def check(run):
    L = box_length(run)
    _, sym0, x0 = next(read_xyz_frames(HERE / "out/start/nma_w216.xyz"))
    nma, xh = bonds(sym0, x0, L)
    heavy = np.array([i for i, s in enumerate(sym0) if s != "H"])
    r0 = np.linalg.norm(mic(x0[nma[:, 1]] - x0[nma[:, 0]], L), axis=1)
    c_ratio = c_xh = b_ratio = b_xh = 0.0
    transfers, n_frames, p = [], 0, 1
    for step, b in frames(run):
        p = len(b)
        cen = b[0] + mic(b - b[0], L).mean(0)
        r = np.linalg.norm(mic(cen[nma[:, 1]] - cen[nma[:, 0]], L), axis=1)
        c_ratio = max(c_ratio, float((r / r0).max()))
        c_xh = max(c_xh, float(np.linalg.norm(mic(cen[xh[:, 1]] - cen[xh[:, 0]], L), axis=1).max()))
        for h, own in zip(xh[:, 1], xh[:, 0]):
            if heavy[np.argmin(np.linalg.norm(mic(cen[heavy] - cen[h], L), axis=1))] != own:
                transfers.append((step, int(h)))
        rb = np.linalg.norm(mic(b[:, nma[:, 1]] - b[:, nma[:, 0]], L), axis=2)
        b_ratio = max(b_ratio, float((rb / r0).max()))
        b_xh = max(b_xh, float(np.linalg.norm(mic(b[:, xh[:, 1]] - b[:, xh[:, 0]], L), axis=2).max()))
        n_frames += 1
    obs = observables(run)
    t, s_t = obs["t_est_k"].mean(), block_sem(obs["t_est_k"])
    d = obs["e_prim_eh"] - obs["e_vir_eh"]
    s_d = block_sem(d) if p > 1 else 0.0
    ok_struct = c_ratio < 1.3 and c_xh < 1.3 and not transfers
    ok_t = abs(t - 300.0 - BIAS_K[p]) < 5 * s_t + 0.5
    ok_e = p == 1 or abs(d.mean() - BIAS_D_EH[p]) < 5 * s_d + 0.25 * abs(BIAS_D_EH[p])
    print(f"{run}: P = {p}, {n_frames} frames, {len(obs)} rows")
    print(f"  centroids: max NMA bond / start {c_ratio:.3f}, max X-H {c_xh:.3f} Å, "
          f"{len(transfers)} transfers {transfers[:5]}: {'ok' if ok_struct else 'FAIL'}")
    print(f"  beads (no limit): max NMA bond / start {b_ratio:.3f}, max X-H {b_xh:.3f} Å")
    print(f"  T_est {t:.2f} K, σ {s_t:.3f} K, offset {t - 300:+.2f} K vs b_P {BIAS_K[p]:+.2f} K: "
          f"{(t - 300 - BIAS_K[p]) / s_t:+.1f}σ: {'ok' if ok_t else 'FAIL'}")
    print(f"  E_prim − E_vir {d.mean():+.3e} Eh vs d_P {BIAS_D_EH[p]:+.2e} Eh, σ_c {s_d:.2e}: "
          f"{'ok' if ok_e else 'FAIL'}")
    return ok_struct and ok_t and ok_e


if __name__ == "__main__":
    sys.exit(0 if all([check(Path(a)) for a in sys.argv[1:]]) else 1)
