"""Regenerate every validation plot and the neon summary from script-produced data (no hand edits).

Run from the repository root after validation/run_neon.py:

    pixi run -e ref python validation/make_plots.py

Inputs: the engine chains in validation/runs/neon_p{1,4,8,16,32}/leg{NN}/ (observables.csv per leg,
rdf.csv cumulative over the chain) and the committed i-PI + LAMMPS reference
crates/pimd-cli/tests/data/ipi_neon.json. The analysis is the one of
crates/pimd-cli/tests/long_neon.rs (WORKFLOW tolerance policy):

- energies: block averages over 250 recorded samples (one 2 500-step leg, 5 ps), checked against
  1 000-sample blocks (sigma_1000 <= 1.5 sigma_250, else the script stops), sigma the larger;
- g(r): per-leg block means from differencing the cumulative rdf.csv of consecutive legs, sigma per
  bin from the 80 one-leg means and the 20 four-leg means, the larger;
- convergence in P: weighted least squares E_vir = E_inf + a / P^2 over P = 4, 8, 16, 32;
- E-1 vs E-2: block errors of the kinetic estimators K = E - V (columns e_*_eh minus v_eh).

Outputs: validation/plots/*.png and validation/plots/neon_summary.json (every number quoted in
validation/RESULTS.md for the M4 neon comparisons), and the same numbers on stdout.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IPI_JSON = ROOT / "crates" / "pimd-cli" / "tests" / "data" / "ipi_neon.json"
K_B = 3.1668115634556e-6  # Eh/K (SPEC U-2, pimd_core::units), for kelvin output only
BOHR_IN_ANGSTROM = 0.529177210903  # CODATA 2018 (pimd_core::units)
LEGS, LEG_STEPS, STRIDE = 80, 2500, 10
BLOCK = LEG_STEPS // STRIDE  # 250 samples per block
HEADER = "step,e_prim_eh,e_vir_eh,v_eh,t_est_k,e_prim_mev,e_vir_mev,e_prim_k,e_vir_k"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (engine / i-PI, P = 16 / P = 1)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def block_mean_sigma(x, n):
    nb = len(x) // n
    assert nb >= 20, f"too few blocks: {nb}"
    means = x[: nb * n].reshape(nb, n).mean(axis=1)
    return means.mean(), means.std(ddof=1) / np.sqrt(nb)


def mean_and_sigma(x, n, what):
    m, s1 = block_mean_sigma(x, n)
    _, s4 = block_mean_sigma(x, 4 * n)
    if not (s4 <= 1.5 * s1 or s1 == s4 == 0.0):
        raise SystemExit(f"{what}: block length {n} too short (sigma {s1:e} vs {s4:e} at 4x)")
    return m, max(s1, s4)


def load_chain(d):
    """(rows [n, 9], r_bohr [bins], g_blocks [LEGS, bins]) of one chain."""
    rows, blocks, prev, r = [], [], None, None
    for leg in range(1, LEGS + 1):
        ld = d / f"leg{leg:02d}"
        lines = (ld / "observables.csv").read_text().splitlines()
        assert lines[0] == HEADER, f"{ld}: unexpected header"
        rows += [[float(v) for v in ln.split(",")] for ln in lines[1:] if ln]
        rdf = np.loadtxt(ld / "rdf.csv", delimiter=",", skiprows=1)
        r, g = rdf[:, 0], rdf[:, 2]
        # Cumulative mean over leg * BLOCK samples -> mean of this leg's samples.
        blocks.append(g if prev is None else leg * g - (leg - 1) * prev)
        prev = g
    rows = np.array(rows)
    assert len(rows) == LEGS * BLOCK and np.all(rows[:, 0] == STRIDE * np.arange(1, len(rows) + 1)), (
        f"{d}: chain rows incomplete or out of order"
    )
    return rows, r, np.array(blocks)


def rdf_stats(blocks):
    m = blocks.mean(axis=0)
    s1 = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    b4 = blocks.reshape(len(blocks) // 4, 4, -1).mean(axis=1)
    s4 = b4.std(axis=0, ddof=1) / np.sqrt(len(b4))
    return m, np.maximum(s1, s4)


def k(x):
    return x / K_B


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "validation" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "plots")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "savefig.dpi": 110})

    ipi = json.loads(IPI_JSON.read_text())
    chains = {p: load_chain(a.runs / f"neon_p{p}") for p in (1, 4, 8, 16, 32)}
    summary = {
        "description": "Liquid-neon PIMD validation (validation/make_plots.py); energies per atom in K",
        "state_point": ipi["system"],
        "block_samples": BLOCK,
        "reference": {"file": "crates/pimd-cli/tests/data/ipi_neon.json", "versions": ipi["versions"]},
    }

    def energy(p, col, what):
        m, s = mean_and_sigma(chains[p][0][:, col], BLOCK, f"P = {p} {what}")
        return k(m), k(s)

    # --- engine vs i-PI at P = 16 -------------------------------------------------------------
    vs = {}
    for col, key in ((2, "e_vir"), (1, "e_prim")):
        m, s = energy(16, col, key)
        mr, sr = k(ipi[key]["mean"]), k(ipi[key]["sigma"])
        sc = np.hypot(s, sr)
        vs[key] = {"engine": [m, s], "ipi": [mr, sr], "diff": m - mr, "diff_sigma_c": (m - mr) / sc}
    _, r, blocks16 = chains[16]
    g16, sg16 = rdf_stats(blocks16)
    gi, si = np.array(ipi["rdf"]["g"]), np.array(ipi["rdf"]["sigma"])
    assert np.allclose(r, ipi["rdf"]["r_bohr"], rtol=1e-12, atol=0), "bin centres differ"
    sc = np.hypot(sg16, si)
    z = np.divide(g16 - gi, sc, out=np.zeros_like(sc), where=sc > 0)
    vs["rdf"] = {"n_bins": len(r), "worst_bin_sigma_c": float(np.abs(z).max()),
                 "worst_bin_r_angstrom": float(r[np.abs(z).argmax()] * BOHR_IN_ANGSTROM),
                 "bins_over_3_sigma_c": int((np.abs(z) > 3).sum())}
    summary["p16_vs_ipi"] = vs

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (key, label) in enumerate((("e_vir", "⟨E_vir⟩"), ("e_prim", "⟨E_prim⟩"))):
        for dx, who, c, name in ((-0.1, "engine", BLUE, "pimd-rs"), (0.1, "ipi", ORANGE, "i-PI + LAMMPS")):
            m, s = vs[key][who]
            ax.errorbar(i + dx, m, yerr=s, fmt="o", ms=6, color=c, capsize=4, lw=2,
                        label=name if i == 0 else None)
    ax.set_xticks([0, 1], ["⟨E_vir⟩", "⟨E_prim⟩"])
    ax.set_xlim(-0.6, 1.6)
    style(ax, "", "energy per atom (K)")
    ax.set_title("Liquid neon, P = 16, T = 30 K: total energy (±1σ, block averages)", color=INK, fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "neon_p16_energy_vs_ipi.png")
    plt.close(fig)

    ra = r * BOHR_IN_ANGSTROM
    fig, (ax, az) = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(ra, gi, color=ORANGE, lw=2, label="i-PI + LAMMPS")
    ax.plot(ra, g16, color=BLUE, lw=2, ls="--", label="pimd-rs")
    style(ax, "", "g(r)")
    ax.set_title("Liquid neon, P = 16, T = 30 K: radial distribution", color=INK, fontsize=9)
    ax.legend(frameon=False)
    az.axhspan(-5, 5, color=GRID, lw=0)
    az.plot(ra, z, "o", ms=3, color=BLUE)
    az.axhline(0, color=MUTED, lw=0.8)
    style(az, "r (Å)", "Δg / σ_c")
    az.set_ylim(-6, 6)
    az.set_yticks([-5, 0, 5])
    fig.tight_layout()
    fig.savefig(a.out / "neon_p16_rdf_vs_ipi.png")
    plt.close(fig)

    # --- quantum (P = 16) vs classical (P = 1) ------------------------------------------------
    g1, sg1 = rdf_stats(chains[1][2])
    b = int(np.argmax(g1))
    qc = {}
    for col, key in ((2, "e_vir"), (1, "e_prim")):
        m1, s1 = energy(1, col, key)
        m16, s16 = energy(16, col, key)
        qc[key] = {"p1": [m1, s1], "p16": [m16, s16], "shift": m16 - m1, "shift_sigma_c": (m16 - m1) / np.hypot(s1, s16)}
    qc["first_peak"] = {"r_angstrom": float(ra[b]), "p1": [float(g1[b]), float(sg1[b])],
                        "p16": [float(g16[b]), float(sg16[b])],
                        "lowering_sigma_c": float((g1[b] - g16[b]) / np.hypot(sg1[b], sg16[b]))}
    summary["quantum_vs_classical"] = qc

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ra, g1, color=ORANGE, lw=2, label="classical, P = 1")
    ax.plot(ra, g16, color=BLUE, lw=2, label="quantum, P = 16")
    style(ax, "r (Å)", "g(r)")
    ax.set_title("Liquid neon, T = 30 K: quantum delocalisation lowers the first peak", color=INK, fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "neon_quantum_vs_classical_rdf.png")
    plt.close(fig)

    # --- convergence in P ---------------------------------------------------------------------
    ps = [4, 8, 16, 32]
    conv = {"points": {}}
    x, y, s = [], [], []
    for p in ps:
        rows = chains[p][0]
        mv, sv = energy(p, 2, "e_vir")
        kp, skp = mean_and_sigma(rows[:, 1] - rows[:, 3], BLOCK, f"P = {p} K_prim")
        kv, skv = mean_and_sigma(rows[:, 2] - rows[:, 3], BLOCK, f"P = {p} K_vir")
        conv["points"][str(p)] = {"e_vir": [mv, sv], "k_prim": [k(kp), k(skp)], "k_vir": [k(kv), k(skv)]}
        x.append(1.0 / p**2)
        y.append(mv)
        s.append(sv)
    x, y, s = map(np.array, (x, y, s))
    w = 1 / s**2
    sw, sx, sy, sxx, sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
    det = sw * sxx - sx * sx
    slope, e_inf = (sw * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det
    res = y - (e_inf + slope * x)
    conv["fit"] = {"e_inf": e_inf, "e_inf_sigma": np.sqrt(sxx / det), "a": slope, "a_sigma": np.sqrt(sw / det),
                   "residual_over_sigma": (res / s).tolist()}
    summary["convergence_in_p"] = conv

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xf = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(xf, e_inf + slope * xf, color=MUTED, lw=1.5,
            label="weighted fit E∞ + a/P²")
    ax.text(0.97, 0.62, f"E∞ = {e_inf:.2f} ± {np.sqrt(sxx / det):.2f} K\na = {slope:.1f} ± {np.sqrt(sw / det):.1f} K",
            transform=ax.transAxes, ha="right", color=INK, fontsize=8)
    ax.errorbar(x, y, yerr=s, fmt="o", ms=6, color=BLUE, capsize=4, lw=2, label="pimd-rs ⟨E_vir⟩ (±1σ)")
    offsets = {4: (-40, -3), 8: (8, -3), 16: (8, 2), 32: (-16, -28)}
    for p, xi, yi in zip(ps, x, y):
        ax.annotate(f"P = {p}", (xi, yi), textcoords="offset points", xytext=offsets[p], color=MUTED, fontsize=8)
    style(ax, "1 / P²", "⟨E_vir⟩ per atom (K)")
    ax.set_title("Liquid neon, T = 30 K: convergence in the number of beads", color=INK, fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out / "neon_convergence_in_p.png")
    plt.close(fig)

    def clean(o):
        if isinstance(o, dict):
            return {kk: clean(v) for kk, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return float(o) if isinstance(o, (np.floating, float)) else o

    (a.out / "neon_summary.json").write_text(json.dumps(clean(summary), indent=1) + "\n")

    for key in ("e_vir", "e_prim"):
        v = vs[key]
        print(f"P = 16 {key}: pimd-rs {v['engine'][0]:.3f} ± {v['engine'][1]:.3f} K, "
              f"i-PI {v['ipi'][0]:.3f} ± {v['ipi'][1]:.3f} K, {v['diff_sigma_c']:+.2f} σ_c")
    print(f"P = 16 g(r): worst bin {vs['rdf']['worst_bin_sigma_c']:.2f} σ_c of {len(r)} "
          f"(r = {vs['rdf']['worst_bin_r_angstrom']:.2f} Å), {vs['rdf']['bins_over_3_sigma_c']} bins > 3 σ_c")
    for key in ("e_vir", "e_prim"):
        q = qc[key]
        print(f"quantum - classical {key}: P = 1 {q['p1'][0]:.3f} ± {q['p1'][1]:.3f}, "
              f"P = 16 {q['p16'][0]:.3f} ± {q['p16'][1]:.3f} K, shift {q['shift']:+.3f} K ({q['shift_sigma_c']:.1f} σ_c)")
    fp = qc["first_peak"]
    print(f"first peak r = {fp['r_angstrom']:.3f} Å: P = 1 {fp['p1'][0]:.4f} ± {fp['p1'][1]:.4f}, "
          f"P = 16 {fp['p16'][0]:.4f} ± {fp['p16'][1]:.4f} ({fp['lowering_sigma_c']:.1f} σ_c)")
    for p in ps:
        c = conv["points"][str(p)]
        print(f"P = {p:2d}: E_vir {c['e_vir'][0]:.3f} ± {c['e_vir'][1]:.3f} K; "
              f"K_prim ± {c['k_prim'][1]:.3f} K, K_vir ± {c['k_vir'][1]:.3f} K")
    f = conv["fit"]
    print(f"fit: E_inf = {f['e_inf']:.3f} ± {f['e_inf_sigma']:.3f} K, a = {f['a']:.1f} ± {f['a_sigma']:.1f} K, "
          f"residuals/σ = {', '.join(f'{v:+.2f}' for v in f['residual_over_sigma'])}")


if __name__ == "__main__":
    main()
