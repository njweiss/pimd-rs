"""Analysis of the M8b runs (docs/experiments/m8-nma-water.md, "Protocol", Analysis): regenerates
every table and figure in studies/m8-nma-water/results/ from out/.

    pixi run -e ref python studies/m8-nma-water/analyse.py [run ...]   # default: every finished run

Runs are named by their output directory under out/ (p1/p1, quantum/p8, ...). Per run, from the
joined parts (health.py): energies per molecule (217 molecules) and T_est from observables.csv;
from every trajectory frame, per bead and then averaged over beads: site RDFs (minimum image,
bins of 0.05 Å on [0, 9) Å; g_OH water–water intermolecular only), hydrogen-bond counts and
H···acceptor distances, NMA bond lengths and torsions (spread of ω about trans). Errors: σ is the largest standard error of
the mean over 1, 2 and 4 ps blocks (rows for energies, frames for structure); a derived quantity
(peak position, coordination number, ratio) is computed per block and its σ taken the same way;
a block where a ratio is undefined (no hydrogen bond in it) is left out of that ratio's σ.
Every scalar is also computed on the two production halves (z = difference / combined σ).
Outputs: summary.json (every scalar of every run), comparisons.csv (P against P = 1, P = 16
against P = 32, the two P = 32 seeds), rdf_<run>.csv, hist_<run>.csv and the figures.
"""

import json
import sys
from pathlib import Path

import numpy as np

from health import HERE, N_NMA, ROWS_PER_PS, bonds, box_length, frames, mic, observables, read_xyz_frames

OUT, RES = HERE / "out", HERE / "results"
RUNS = ["p1/p1", "quantum/p8", "quantum/p16", "quantum/p32", "quantum/p32_seed2"]
N_MOL = 217
FRAMES_PER_PS = 50  # trajectory every 40 steps of 0.5 fs
BLOCK_PS = (1, 2, 4)
BINS = np.arange(0.0, 9.0 + 1e-9, 0.05)
R_MID = 0.5 * (BINS[1:] + BINS[:-1])
SHELL = 4.0 / 3.0 * np.pi * (BINS[1:] ** 3 - BINS[:-1] ** 3)
HB_CUTS = [(3.5, 30.0), (3.3, 30.0), (3.7, 30.0), (3.5, 20.0), (3.5, 40.0)]  # first = reported
HIST = {"r_co": (1.1, 1.4), "r_cn": (1.2, 1.5), "r_nh": (0.8, 1.3), "omega": (-180, 180), "ocnh": (-180, 180)}
HIST_BINS = 120


def sites():
    """NMA site indices from the start structure's bonds, and the water O and H indices."""
    _, sym, x0 = next(read_xyz_frames(HERE / "out/start/nma_w216.xyz"))
    L = box_length(OUT / "p1/p1")
    nma, _ = bonds(sym, x0, L)
    nb = {i: set() for i in range(N_NMA)}
    for i, j in nma:
        nb[i].add(j)
        nb[j].add(i)
    o = next(i for i in range(N_NMA) if sym[i] == "O")
    n = next(i for i in range(N_NMA) if sym[i] == "N")
    c = next(j for j in nb[o] if sym[j] == "C")
    hn = next(j for j in nb[n] if sym[j] == "H")
    cm1 = next(j for j in nb[c] if sym[j] == "C")
    cm2 = next(j for j in nb[n] if sym[j] == "C" and j != c)
    ow = np.arange(N_NMA, len(sym), 3)
    return dict(O=o, N=n, C=c, HN=hn, CM1=cm1, CM2=cm2), ow, np.stack([ow + 1, ow + 2], 1)


def dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def angle_deg(u, v):
    cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def per_bead(b, s, ow, hw, L):
    """Per-bead quantities: histogram counts of the six RDFs, H-bond counts and distance sums for
    every cutoff pair, and the NMA geometry."""
    out = {}
    xo, xw, xh = b[s["O"]], b[ow], b[hw]                       # xh: [216, 2, 3]

    def h(x0, xs):
        r = np.linalg.norm(mic(xs - x0, L), axis=-1)
        return np.histogram(r, BINS)[0]

    out["g_onma_hw"] = h(xo, xh.reshape(-1, 3))
    out["g_onma_ow"] = h(xo, xw)
    out["g_hn_ow"] = h(b[s["HN"]], xw)
    out["g_n_ow"] = h(b[s["N"]], xw)
    doo = np.linalg.norm(mic(xw[None] - xw[:, None], L), axis=-1)
    out["g_oo"] = np.histogram(doo[np.triu_indices(len(ow), 1)], BINS)[0] * 2
    doh = np.linalg.norm(mic(xh.reshape(1, -1, 3) - xw[:, None], L), axis=-1)
    own = np.repeat(np.arange(len(ow)), 2)
    doh[own, np.arange(len(own))] = np.inf
    out["g_oh"] = np.histogram(doh[np.isfinite(doh)], BINS)[0]
    # C=O···H–O_w: r(O_NMA···O_w) and ∠(O_w→H_w, O_w→O_NMA); N–H···O_w: r(N···O_w), ∠(N→H, N→O_w).
    v_wo = mic(xo - xw, L)                                     # O_w → O_NMA
    r_wo = np.linalg.norm(v_wo, axis=-1)
    v_wh = mic(xh - xw[:, None], L)                            # O_w → H_w
    a_co = angle_deg(v_wh, v_wo[:, None])                      # [216, 2]
    d_co = np.linalg.norm(mic(xo - xh, L), axis=-1)            # H_w ··· O_NMA
    v_nw = mic(xw - b[s["N"]], L)
    r_nw = np.linalg.norm(v_nw, axis=-1)
    a_nh = angle_deg(mic(b[s["HN"]] - b[s["N"]], L)[None], v_nw)
    d_nh = np.linalg.norm(mic(xw - b[s["HN"]], L), axis=-1)    # H_N ··· O_w
    for k, (rc, ac) in enumerate(HB_CUTS):
        m_co = (r_wo[:, None] < rc) & (a_co < ac)
        m_nh = (r_nw < rc) & (a_nh < ac)
        out[f"hb_co_n{k}"], out[f"hb_co_d{k}"] = m_co.sum(), d_co[m_co].sum()
        out[f"hb_nh_n{k}"], out[f"hb_nh_d{k}"] = m_nh.sum(), d_nh[m_nh].sum()
    dist = lambda i, j: np.linalg.norm(mic(b[j] - b[i], L))
    out["r_co"], out["r_cn"], out["r_nh"] = dist(s["C"], s["O"]), dist(s["C"], s["N"]), dist(s["N"], s["HN"])
    out["r_nh2"] = out["r_nh"] ** 2
    # Torsions from unwrapped NMA positions relative to C (one molecule, minimum image per atom).
    p = {k: b[s["C"]] + mic(b[i] - b[s["C"]], L) for k, i in s.items()}
    out["omega"] = dihedral(p["CM1"], p["C"], p["N"], p["CM2"])
    out["ocnh"] = dihedral(p["O"], p["C"], p["N"], p["HN"])
    out["trans"] = float(abs(out["omega"]) > 150.0)
    dev = (out["omega"] % 360.0) - 180.0                        # signed deviation from trans
    out["omega_dev"], out["omega_dev2"] = dev, dev * dev
    return out


def frame_series(run, s, ow, hw):
    """Per-frame, bead-averaged arrays of every per-bead quantity (histograms of the geometry
    added), keyed by name, each [n_frames, ...]."""
    L = box_length(run)
    rows = []
    for _, beads in frames(run):
        per = [per_bead(b, s, ow, hw, L) for b in beads]
        f = {k: np.mean([q[k] for q in per], 0) for k in per[0]}
        for k, (lo, hi) in HIST.items():
            f[f"h_{k}"] = np.histogram([q[k] for q in per], HIST_BINS, (lo, hi))[0] / len(per)
        rows.append(f)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}, len(beads), L


def blocks(x, per_ps, ps):
    b = ps * per_ps
    m = len(x) // b
    return x[: m * b].reshape(m, b, *x.shape[1:]).mean(1)


def stat(series, fn, per_ps):
    """fn(means) over the whole series, σ the largest block SE of fn over 1, 2, 4 ps blocks;
    the same for each half, and z of the halves."""

    def one(x):
        v = fn(*[a.mean(0) for a in x])
        s = []
        for ps in BLOCK_PS:
            bl = [blocks(a, per_ps, ps) for a in x]
            if len(bl[0]) < 2:
                continue
            vals = np.array([fn(*[a[i] for a in bl]) for i in range(len(bl[0]))])
            vals = vals[np.isfinite(vals)]  # a ratio over a block without any event
            s.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        return float(v), float(max(s))

    v, s = one(series)
    n = len(series[0]) // 2
    (v1, s1), (v2, s2) = one([a[:n] for a in series]), one([a[n:] for a in series])
    return {"v": v, "s": s, "h1": v1, "s1": s1, "h2": v2, "s2": s2, "z_halves": (v2 - v1) / np.hypot(s1, s2)}


def rdf_norm(name, L):
    """Pairs per bin of an ideal gas: n_A · n_B / V · shell volume."""
    v = L ** 3
    n = {"g_onma_hw": (1, 432), "g_onma_ow": (1, 216), "g_hn_ow": (1, 216), "g_n_ow": (1, 216),
         "g_oo": (216, 215), "g_oh": (216, 430)}[name]
    return n[0] * n[1] / v * SHELL, n[1] / v


# First-peak search windows (Å): the hydrogen-bond shell of each pair. The single-site RDFs are
# noisy and their first peak can be lower than the bulk shells beyond it, so the global maximum
# is not used (P = 1 test of this script).
# Coordination-number cutoffs (Å), the same for every run so that runs compare: the P = 1 first
# minima rounded to 0.05 Å (water O–O 3.55, O–H 2.45 the usual choices). A run's own first
# minimum is reported but moves by up to 0.2 Å between runs in the noisy single-site RDFs.
CN_CUT = {"g_onma_hw": 2.6, "g_onma_ow": 3.5, "g_hn_ow": 2.6, "g_n_ow": 3.4, "g_oo": 3.55, "g_oh": 2.45}
PEAK_WINDOW = {"g_onma_hw": (1.4, 2.6), "g_onma_ow": (2.4, 3.4), "g_hn_ow": (1.5, 2.6),
               "g_n_ow": (2.6, 3.5), "g_oo": (2.4, 3.2), "g_oh": (1.4, 2.4)}


def peak(g, window):
    """First-peak position (parabola through the highest bin in the window and its neighbours),
    height, and first minimum (lowest bin within 1 Å past the peak bin)."""
    lo, hi = np.searchsorted(R_MID, window)
    i = lo + int(np.argmax(g[lo:hi]))
    y0, y1, y2 = g[i - 1], g[i], g[i + 1]
    den = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    r_pk = R_MID[i] + dx * (BINS[1] - BINS[0])
    j = i + int(np.argmin(g[i:i + 20]))
    return r_pk, y1 - 0.25 * (y0 - y2) * dx, R_MID[j]


def analyse(name):
    run = OUT / name
    s, ow, hw = sites()
    obs = observables(run)
    res = {}
    to_mol = 660.0 / N_MOL
    for k, col in (("e_vir_eh_per_mol", "e_vir_eh"), ("e_prim_eh_per_mol", "e_prim_eh"), ("v_eh_per_mol", "v_eh")):
        res[k] = stat([obs[col] * to_mol], lambda a: a, ROWS_PER_PS)
    res["t_est_k"] = stat([obs["t_est_k"]], lambda a: a, ROWS_PER_PS)
    fs, p, L = frame_series(run, s, ow, hw)
    res["P"], res["n_frames"], res["n_rows"] = p, len(fs["r_nh"]), len(obs)
    rdf = {}
    for g in ("g_onma_hw", "g_onma_ow", "g_hn_ow", "g_n_ow", "g_oo", "g_oh"):
        norm, rho = rdf_norm(g, L)
        full = fs[g].mean(0) / norm
        rdf[g] = full
        w = PEAK_WINDOW[g]
        r_min = peak(full, w)[2]
        cut = R_MID < CN_CUT[g]
        res[f"{g}_peak_r"] = stat([fs[g]], lambda a: peak(a / norm, w)[0], FRAMES_PER_PS)
        res[f"{g}_peak_h"] = stat([fs[g]], lambda a: peak(a / norm, w)[1], FRAMES_PER_PS)
        res[f"{g}_r_min"] = r_min
        # Coordination number: pairs per A-site inside CN_CUT.
        n_a = norm[0] / (rho * SHELL[0])
        res[f"{g}_cn"] = stat([fs[g]], lambda a: a[cut].sum() / n_a, FRAMES_PER_PS)
    for k, (rc, ac) in enumerate(HB_CUTS):
        tag = "" if k == 0 else f"_{rc}A_{ac:.0f}deg"
        for d in ("co", "nh"):
            res[f"hb_{d}_count{tag}"] = stat([fs[f"hb_{d}_n{k}"]], lambda a: a, FRAMES_PER_PS)
            if k == 0:
                res[f"hb_{d}_dist"] = stat([fs[f"hb_{d}_d0"], fs[f"hb_{d}_n0"]], lambda a, b: a / b if b > 0 else np.nan, FRAMES_PER_PS)
    for k in ("r_co", "r_cn", "r_nh", "trans"):
        res[k] = stat([fs[k]], lambda a: a, FRAMES_PER_PS)
    res["omega_sd_deg"] = stat([fs["omega_dev"], fs["omega_dev2"]], lambda a, b: np.sqrt(max(b - a * a, 0.0)),
                               FRAMES_PER_PS)
    res["r_nh_sd"] = stat([fs["r_nh"], fs["r_nh2"]], lambda a, b: np.sqrt(max(b - a * a, 0.0)), FRAMES_PER_PS)
    res["throughput_ms_per_step"] = throughput(run)
    hist = {k: fs[f"h_{k}"].mean(0) for k in HIST}
    return res, rdf, hist


def throughput(run):
    """ms per step of the last invocation (startup included): all steps for a fresh run, the
    production steps after the checkpoint for a restarted one."""
    cfg = dict(l.split(" = ") for l in (run / "config.toml").read_text().splitlines() if " = " in l)
    n_equil, n_steps = int(cfg["n_equil"]), int(cfg["n_steps"])
    wall = float((run / "wall.txt").read_text().split()[-5])
    first = int(np.genfromtxt(run / "observables.csv", delimiter=",", names=True, max_rows=1)["step"])
    steps = n_equil + n_steps if not list(run.glob("part*")) else n_steps - first + 1
    return 1000.0 * wall / steps


def comparisons(summary):
    pairs = [("quantum/p8", "p1/p1"), ("quantum/p16", "p1/p1"), ("quantum/p32", "p1/p1"),
             ("quantum/p32", "quantum/p16"), ("quantum/p32_seed2", "quantum/p32")]
    lines = ["a,b,observable,a_value,b_value,diff,sigma_c,z"]
    for a, b in pairs:
        if a not in summary or b not in summary:
            continue
        for k, va in summary[a].items():
            vb = summary[b].get(k)
            if not isinstance(va, dict) or not isinstance(vb, dict):
                continue
            d, sc = va["v"] - vb["v"], np.hypot(va["s"], vb["s"])
            lines.append(f"{a},{b},{k},{va['v']:.8g},{vb['v']:.8g},{d:.6g},{sc:.3g},{d / sc:.2f}")
    (RES / "comparisons.csv").write_text("\n".join(lines) + "\n")


def figures(summary, rdfs, hists):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [r for r in ("p1/p1", "quantum/p32") if r in rdfs]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, g in zip(axs.flat, ("g_onma_hw", "g_onma_ow", "g_hn_ow", "g_n_ow", "g_oo", "g_oh")):
        for r in have:
            ax.plot(R_MID, rdfs[r][g], label=f"P = {summary[r]['P']}")
        ax.set_title(g)
        ax.set_xlim(1.0, 7.0)
    axs[1, 0].set_xlabel("r (Å)")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(RES / "rdf_p1_p32.png", dpi=120)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for r, h in hists.items():
        for ax, k in zip(axs, ("r_nh", "omega")):
            lo, hi = HIST[k]
            x = np.linspace(lo, hi, HIST_BINS + 1)
            ax.plot(0.5 * (x[1:] + x[:-1]), h[k] / (h[k].sum() * (x[1] - x[0])), label=r.split("/")[1])
            ax.set_xlabel({"r_nh": "r(N–H) (Å)", "omega": "ω (deg)"}[k])
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(RES / "nh_omega_vs_p.png", dpi=120)
    fig, ax = plt.subplots(figsize=(5, 4))
    for r, v in summary.items():
        e = v["e_vir_eh_per_mol"]
        ax.errorbar(1.0 / v["P"] ** 2, e["v"], e["s"], fmt="o", label=r.split("/")[1])
    ax.set_xlabel("1/P²")
    ax.set_ylabel("⟨E_vir⟩ (Eh per molecule)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "evir_vs_invp2.png", dpi=120)


def main():
    names = sys.argv[1:] or [r for r in RUNS if (OUT / r / "wall.txt").exists()]
    RES.mkdir(exist_ok=True)
    old = RES / "summary.json"
    summary = json.loads(old.read_text()) if old.exists() else {}
    rdfs, hists = {}, {}
    for name in names:
        res, rdf, hist = analyse(name)
        summary[name] = res
        tag = name.split("/")[1]
        np.savetxt(RES / f"rdf_{tag}.csv", np.column_stack([R_MID] + list(rdf.values())), delimiter=",",
                   header="r_angstrom," + ",".join(rdf), comments="", fmt="%.6g")
        cols = [np.linspace(lo, hi, HIST_BINS + 1)[:-1] for lo, hi in HIST.values()]
        np.savetxt(RES / f"hist_{tag}.csv", np.column_stack([c for pair in zip(cols, hist.values()) for c in pair]),
                   delimiter=",", header=",".join(f"{k}_lo,{k}_frac" for k in HIST), comments="", fmt="%.6g")
        print(name, "done", flush=True)
    for name in summary:
        tag = name.split("/")[1]
        if name not in rdfs and (RES / f"rdf_{tag}.csv").exists():
            a = np.genfromtxt(RES / f"rdf_{tag}.csv", delimiter=",", names=True)
            rdfs[name] = {k: a[k] for k in a.dtype.names[1:]}
            b = np.genfromtxt(RES / f"hist_{tag}.csv", delimiter=",", names=True)
            hists[name] = {k: b[f"{k}_frac"] for k in HIST}
    old.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    comparisons(summary)
    figures(summary, rdfs, hists)


if __name__ == "__main__":
    main()
