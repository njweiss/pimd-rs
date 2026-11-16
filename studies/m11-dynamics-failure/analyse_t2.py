"""M11 tier 2: centroid velocity spectra of gas-phase H₂O against the surface's band origins.

    pixi run -e ref python studies/m11-dynamics-failure/analyse_t2.py RUN_DIR [RUN_DIR ...] [--json OUT]

RUN_DIRs with the same (T, method, P, Δt, Ω, client route) are replicas; their C_vv are averaged and
σ of each metric is the standard error over replicas. The transform is tier 1's (analyse.spectrum:
Hann window over the full lag range, trapezoid rule). Windows and references come from
results/tier2/reference.json and were fixed before any dynamics run (plan "Analysis"):
- bend: ν2 ± 300 cm⁻¹; stretch: ν1 − 300 to ν3 + 300 cm⁻¹. Per band: first-moment centre,
  standard-deviation width, peak (three-bin parabola, unrefined on a window edge).
- references: bend ν2; stretch the Y-4-weighted mean of ν1 and ν3 (and ν1, ν3 themselves).
- full range (ω ≥ 1000 cm⁻¹): the first moment of each half (below and above 2500 cm⁻¹), and the
  share of the weight inside each band window.
- spurious: fraction of ∫|I| over ω ≥ 1000 cm⁻¹ more than 300 cm⁻¹ from every one of ν2, ν1, ν3,
  2ν2.
- hole check: the lowest per-molecule potential in observables.csv against E_min (a molecule in
  one of the surface's holes would push it hundreds of Eh down).
"""

import argparse
import json
import os
import sys
import tomllib
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse as A  # noqa: E402

CM = A.CM
FS = A.FS


def reference():
    with open(os.path.join(HERE, "results/tier2/reference.json")) as f:
        r = json.load(f)
    lv = {t["n"]: t["E_cm"] for t in r["levels"]}
    w = r["y4_weights"]
    ref = {"nu2": lv[1], "2nu2": lv[2], "nu1": lv[3], "nu3": lv[4],
           "stretch": (w[1] * lv[3] + w[2] * lv[4]) / (w[1] + w[2]), "E_min": r["E_min_eh"]}
    ref["windows"] = {"bend": (lv[1] - 300, lv[1] + 300), "stretch": (lv[3] - 300, lv[4] + 300)}
    return ref


def band(cm, i_w, lo, hi):
    win = (cm >= lo) & (cm <= hi)
    iw = np.clip(i_w[win], 0.0, None)
    centre = float((cm[win] * iw).sum() / iw.sum())
    width = float(np.sqrt(((cm[win] - centre) ** 2 * iw).sum() / iw.sum()))
    k = int(np.argmax(i_w[win]))
    if 0 < k < win.sum() - 1:
        y0, y1, y2 = i_w[win][k - 1:k + 2]
        peak = float(cm[win][k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    else:
        peak = float(cm[win][k])
    return centre, width, peak


def metrics(i_w, ref):
    cm = A.OMEGA * CM
    out = {}
    for name, (lo, hi) in ref["windows"].items():
        out[f"{name}_centre"], out[f"{name}_width"], out[f"{name}_peak"] = band(cm, i_w, lo, hi)
    hi_ = cm >= 1000
    ip = np.clip(i_w, 0.0, None)
    tot = np.trapezoid(ip * hi_, cm)
    for name, sel in (("low", hi_ & (cm < 2500)), ("high", cm >= 2500)):
        out[f"{name}_centre_full"] = float(np.trapezoid(cm * ip * sel, cm) / np.trapezoid(ip * sel, cm))
    for name, (lo, hi) in ref["windows"].items():
        out[f"{name}_share"] = float(np.trapezoid(ip * ((cm >= lo) & (cm <= hi)), cm) / tot)
    far = hi_.copy()
    for k in ("nu2", "nu1", "nu3", "2nu2"):
        far &= np.abs(cm - ref[k]) > 300
    out["spurious"] = float(np.trapezoid(np.abs(i_w) * far, cm) / np.trapezoid(np.abs(i_w) * hi_, cm))
    return out


def load(d):
    with open(os.path.join(d, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    c = np.genfromtxt(os.path.join(d, "correlations.csv"), delimiter=",", names=True)
    obs = np.genfromtxt(os.path.join(d, "observables.csv"), delimiter=",", names=True)
    return cfg, c, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--route", action="append", default=[],
                    help="DIR=LABEL: the client route of a run (default: f64 if the directory name says so, else cueq)")
    a = ap.parse_args()
    ref = reference()
    routes = dict(r.split("=", 1) for r in a.route)
    groups = defaultdict(list)
    for d in a.runs:
        cfg, c, obs = load(d)
        route = routes.get(d, "f64" if "f64" in os.path.basename(os.path.normpath(d)) else "cueq")
        s = cfg["sampling"]
        dyn = cfg.get("dynamics", {})
        method = "classical" if s["n_beads"] == 1 else dyn.get("method", "static")
        key = (s["temperature_kelvin"], method, s["n_beads"], s["dt_fs"], dyn.get("cmd_omega_cm"), route)
        groups[key].append((d, c, obs))
    rows = []
    for key, reps in sorted(groups.items(), key=lambda kv: str(kv[0])):
        tk, method, p, dt, omega, route = key
        t = reps[0][1]["t_fs"] * FS
        per = [metrics(A.spectrum(t, c["c_vv_au"]), ref) for _, c, _ in reps]
        m = metrics(A.spectrum(t, np.mean([c["c_vv_au"] for _, c, _ in reps], 0)), ref)
        vmin = min(float(3 * o["v_eh"].min()) for _, _, o in reps) - ref["E_min"]
        row = {"T": tk, "method": method, "P": p, "dt_fs": dt, "omega_cm": omega, "route": route,
               "replicas": len(reps), "dirs": [d for d, _, _ in reps], **m,
               **{f"{k}_sigma": float(np.std([q[k] for q in per], ddof=1) / np.sqrt(len(per)))
                  if len(per) > 1 else None for k in m},
               "v_min_above_emin_eh_per_molecule": vmin}
        rows.append(row)
        s = lambda k: f"{row[k]:7.1f}" + (f"±{row[k + '_sigma']:.1f}" if row.get(k + "_sigma") else "")
        print(f"{tk:4.0f}K {method:9s} P={p:<3d} dt={dt} Ω={omega} {route} n={len(reps)}: "
              f"bend {s('bend_centre')} (ν2 {ref['nu2']:.1f}) w {s('bend_width')} pk {s('bend_peak')}; "
              f"stretch {s('stretch_centre')} (ref {ref['stretch']:.1f}) w {s('stretch_width')} pk {s('stretch_peak')}; "
              f"full {s('low_centre_full')} / {s('high_centre_full')}; shares {row['bend_share']:.2f}/{row['stretch_share']:.2f}; "
              f"spurious {row['spurious']:.3f}; min V−E_min per molecule {vmin:+.2e} Eh")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"reference": ref, "groups": rows}, f, indent=1)


if __name__ == "__main__":
    main()
