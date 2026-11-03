"""Tables and figures of the M6c study from the production runs (docs/experiments/m6-water-mace.md,
"Protocol", analysis).

    pixi run -e ref python studies/m6-water-mace/analyse.py

Reads studies/m6-water-mace/out/production/<run>/{observables.csv,trajectory.xyz} and
results/harmonic_{mace,xtb}.json; writes results/summary.json, results/runs.csv and
results/fig_*.png. Runs that are missing or incomplete are skipped with a message.

Per run (per molecule; energies minus the model's V_min): ⟨E_vir⟩, ⟨E_prim⟩, ⟨V̄⟩ from every
production step; bead-averaged ⟨r_OH⟩ (for HDO: ⟨r_OD⟩, atom 1, and ⟨r_OH⟩, atom 2, separately)
and ⟨θ_HOH⟩ from the trajectory (every 10 steps). σ by block averaging (WORKFLOW): blocks of 2 ps
and 8 ps, the larger σ.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "production"
RES = HERE / "results"
BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2
K_B = 3.1668115634556e-6  # Eh/K
DT_FS, TRAJ_EVERY = 0.5, 10
BLOCK_PS = 2.0
HIST = (1.4, 2.4, 25)  # bohr, as M5
# M5 GFN2-xTB, P = 16, pimd-rs (validation/RESULTS.md, "Water on GFN2-xTB"): raw energies, Eh.
M5_XTB_P16 = {"e_vir": (-5.049908, 0.000023), "e_prim": (-5.049985, 0.000047), "v": (-5.061468, 0.000017),
              "r_oh": (1.849247, 0.000422), "theta": (107.0878, 0.0149)}

RUNS = {  # name: (species, model, P)
    **{f"h2o_mace_p{p}": ("h2o", "mace", p) for p in (1, 4, 8, 16, 32)},
    "h2o_mace_p32_seed2": ("h2o", "mace", 32),
    "h2o_xtb_p1": ("h2o", "xtb", 1), "h2o_xtb_p32": ("h2o", "xtb", 32),
    **{f"{s}_mace_p{p}": (s, "mace", p) for s in ("d2o", "hdo") for p in (1, 32)},
}


def block_sigma(x, block):
    """Standard error of the mean from blocks of `block` and 4·`block` samples, the larger."""
    out = []
    for b in (block, 4 * block):
        n = len(x) // b
        if n < 2:
            continue
        m = x[: n * b].reshape(n, b).mean(axis=1)
        out.append(m.std(ddof=1) / math.sqrt(n))
    return max(out)


def stat(x, block):
    x = np.asarray(x)
    return [float(x.mean()), float(block_sigma(x, block))]


def read_traj(path, p):
    """Positions in bohr, shape [frames, P, 3 atoms, 3]."""
    lines = path.read_text().split("\n")
    stride = 2 + 3 * p
    nf = len([ln for ln in lines if ln.strip()]) // stride
    xyz = np.empty((nf, p * 3, 3))
    for f in range(nf):
        block = lines[f * stride + 2 : (f + 1) * stride]
        xyz[f] = [[float(v) for v in ln.split()[1:4]] for ln in block]
    return xyz.reshape(nf, p, 3, 3) / BOHR_IN_ANGSTROM


def analyse_run(name, species, model, p, vmin):
    d = OUT / name
    obs = d / "observables.csv"
    if not obs.exists() or not (d / "wall.txt").exists():
        print(f"skip {name}: not finished")
        return None
    rows = list(csv.DictReader(obs.open()))
    col = lambda k: np.array([3.0 * float(r[k]) for r in rows])  # per atom → per molecule
    block = int(BLOCK_PS * 1000 / DT_FS)
    res = {"species": species, "model": model, "p": p, "n_steps": len(rows),
           "e_vir": stat(col("e_vir_eh") - vmin, block), "e_prim": stat(col("e_prim_eh") - vmin, block),
           "v": stat(col("v_eh") - vmin, block),
           "e_vir_minus_e_prim": stat(col("e_vir_eh") - col("e_prim_eh"), block),
           "t_est_k": stat([float(r["t_est_k"]) for r in rows], block)}
    x = read_traj(d / "trajectory.xyz", p)
    b1, b2 = x[:, :, 1] - x[:, :, 0], x[:, :, 2] - x[:, :, 0]
    r1, r2 = np.linalg.norm(b1, axis=2), np.linalg.norm(b2, axis=2)  # [frames, P]
    theta = np.degrees(np.arccos(np.clip((b1 * b2).sum(axis=2) / (r1 * r2), -1, 1))).mean(axis=1)
    fb = block // TRAJ_EVERY
    if species == "hdo":
        res["r_od"] = stat(r1.mean(axis=1), fb)
        res["r_oh"] = stat(r2.mean(axis=1), fb)
        res["r_oh_minus_r_od"] = stat(r2.mean(axis=1) - r1.mean(axis=1), fb)
        hist_r = r2
    else:
        res["r_oh"] = stat(0.5 * (r1 + r2).mean(axis=1), fb)
        hist_r = np.concatenate([r1, r2], axis=1)
    res["theta"] = stat(theta, fb)
    lo, hi, nb = HIST
    h, edges = np.histogram(hist_r.ravel(), bins=nb, range=(lo, hi), density=False)
    res["r_hist"] = {"edges_bohr": edges.tolist(), "density": (h / (hist_r.size * (hi - lo) / nb)).tolist()}
    res["frames"] = int(x.shape[0])
    return res


def diff(a, b):
    """a − b with σ combined in quadrature, and the difference in units of σ_c."""
    d, s = a[0] - b[0], math.hypot(a[1], b[1])
    return [d, s, d / s if s > 0 else float("inf")]


def main():
    harm = {m: json.loads((RES / f"harmonic_{m}.json").read_text()) for m in ("mace", "xtb")}
    runs = {}
    for name, (sp, model, p) in RUNS.items():
        r = analyse_run(name, sp, model, p, harm[model]["v_min_eh"])
        if r:
            r["harmonic_e"] = harm[model]["isotopes"][sp]["e_minus_vmin_eh"][str(p)]
            r["e_vir_minus_harmonic"] = r["e_vir"][0] - r["harmonic_e"]
            runs[name] = r

    checks, comp = {}, {}
    for name, r in runs.items():
        # One trajectory: σ from the blocks of the difference series (the estimators are correlated).
        m, sd = r["e_vir_minus_e_prim"]
        checks[f"{name}: e_vir − e_prim"] = [m, sd, m / sd if sd > 0 else 0.0]
    if {"h2o_mace_p32", "h2o_mace_p32_seed2"} <= runs.keys():
        for k in ("e_vir", "r_oh", "theta"):
            checks[f"h2o_mace_p32 seed 1 − seed 2: {k}"] = diff(runs["h2o_mace_p32"][k], runs["h2o_mace_p32_seed2"][k])
    for sp, model in (("h2o", "mace"), ("h2o", "xtb"), ("d2o", "mace"), ("hdo", "mace")):
        a, b = runs.get(f"{sp}_{model}_p32"), runs.get(f"{sp}_{model}_p1")
        if a and b:
            q = {k: diff(a[k], b[k]) for k in ("e_vir", "e_prim", "v", "r_oh", "theta") if k in a}
            if sp == "hdo":
                q["r_od"] = diff(a["r_od"], b["r_od"])
            q["e_vir_harmonic"] = a["harmonic_e"] - b["harmonic_e"]
            comp[f"{sp} {model}: P = 32 − P = 1"] = q
    h, d = comp.get("h2o mace: P = 32 − P = 1"), comp.get("d2o mace: P = 32 − P = 1")
    if h and d:
        comp["d2o/h2o ratio of quantum−classical e_vir (mace)"] = d["e_vir"][0] / h["e_vir"][0]
    if "h2o_mace_p32" in runs and "h2o_xtb_p32" in runs:
        comp["mace − xtb at P = 32"] = {k: diff(runs["h2o_mace_p32"][k], runs["h2o_xtb_p32"][k]) for k in ("e_vir", "r_oh", "theta")}
    if "h2o_xtb_p1" in runs:  # M5 run at P = 16 in the same frame (E − V_min)
        vx = harm["xtb"]["v_min_eh"]
        comp["xtb M5 P = 16 (E − V_min)"] = {"e_vir": [M5_XTB_P16["e_vir"][0] - vx, M5_XTB_P16["e_vir"][1]],
                                             "r_oh": list(M5_XTB_P16["r_oh"]), "theta": list(M5_XTB_P16["theta"]),
                                             "harmonic_e": harm["xtb"]["isotopes"]["h2o"]["e_minus_vmin_eh"]["16"]}

    RES.mkdir(exist_ok=True)
    (RES / "summary.json").write_text(json.dumps({"runs": {k: {kk: vv for kk, vv in v.items() if kk != "r_hist"} for k, v in runs.items()},
                                                  "checks": checks, "comparisons": comp}, indent=1) + "\n")
    with (RES / "runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "P", "steps", "T_est_K", "E_vir-Vmin_Eh", "s", "E_prim-Vmin_Eh", "s", "V-Vmin_Eh", "s",
                    "harmonic_Eh", "r_OH_bohr", "s", "theta_deg", "s", "r_OD_bohr", "s"])
        for k, r in runs.items():
            w.writerow([k, r["p"], r["n_steps"], "%.1f" % r["t_est_k"][0],
                        *["%.6f" % v for q in ("e_vir", "e_prim", "v") for v in r[q]], "%.6f" % r["harmonic_e"],
                        *["%.5f" % v for v in r["r_oh"]], *["%.3f" % v for v in r["theta"]],
                        *(["%.5f" % v for v in r["r_od"]] if "r_od" in r else ["", ""])])
    figures(runs, harm)
    for k, v in checks.items():
        print(f"check {k}: {v[0]:+.2e} ± {v[1]:.1e} ({v[2]:+.2f} σ)")
    print(json.dumps(comp, indent=1))


def figures(runs, harm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, colour in (("mace", "C0"), ("xtb", "C1")):
        hp = harm[model]["isotopes"]["h2o"]["e_minus_vmin_eh"]
        ps = [int(p) for p in hp if p != "inf"]
        ax.plot([1 / p**2 for p in ps], [hp[str(p)] for p in ps], "-", color=colour, lw=1, label=f"{model} harmonic")
        pts = [(r["p"], r["e_vir"]) for k, r in runs.items() if r["model"] == model and r["species"] == "h2o" and "seed2" not in k]
        if pts:
            ax.errorbar([1 / p**2 for p, _ in pts], [e[0] for _, e in pts], [e[1] for _, e in pts], fmt="o", color=colour, label=f"{model} PIMD E_vir")
    ax.set_xscale("log")
    ax.set_xlabel("1/P²")
    ax.set_ylabel("⟨E⟩ − V_min (Eh)")
    ax.set_title("H₂O, 300 K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "fig_energy_vs_p.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, style in (("h2o_mace_p1", "C0--"), ("h2o_mace_p32", "C0-"), ("h2o_xtb_p1", "C1--"), ("h2o_xtb_p32", "C1-")):
        if k in runs:
            e, dens = runs[k]["r_hist"]["edges_bohr"], runs[k]["r_hist"]["density"]
            ax.plot([(a + b) / 2 for a, b in zip(e, e[1:])], dens, style, label=k)
    ax.set_xlabel("r_OH (bohr)")
    ax.set_ylabel("density (1/bohr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "fig_r_oh.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
