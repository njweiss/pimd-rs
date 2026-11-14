"""M11 tier 1: centroid velocity spectra and errors against the exact reference.

    pixi run -e ref python studies/m11-dynamics-failure/analyse.py RUN_DIR [RUN_DIR ...] [--json OUT]

Each RUN_DIR holds a finished run (config.toml, correlations.csv); RUN_DIRs given together with the
same system, T and method are replicas (independent seeds): their C_vv are averaged with equal
weights, and the σ of each spectral metric is the standard error over replicas. Per group:
- I(ω) = ∫_0^T C_vv(t) w(t) cos(ωt) dt, Hann window w(t) = cos²(πt/(2T)) over the full lag range,
  trapezoid rule; the exact C_vv is sampled on the same lag grid and transformed identically.
- band centre and width: first moment and standard deviation of I over the stretch window
  2800–4400 cm⁻¹; peak: maximum in that window, refined by a parabola through three bins.
- spurious weight: ∫|I − I_exact| outside ±150 cm⁻¹ of the exact peak, over ∫ I_exact (all ω ≥ 0).
- time-domain error: max over 0–500 fs of |C_vv − C_exact| / C_exact(0), and the same in units of
  the engine's σ (c_vv_sem_au combined over replicas).
"""

import argparse
import json
import os
import sys
import tomllib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact  # noqa: E402

FS = 41.341373336561  # ħ/Eh per fs, U-2
CM = 219474.6313632  # cm⁻¹ per Eh, U-2
OMEGA = np.arange(0.0, 6000.0, 1.0) / CM  # Eh


def method_of(cfg):
    dyn = cfg.get("dynamics", {})
    m = dyn.get("method", "static")
    if m == "rpmd" and cfg["sampling"]["n_beads"] == 1:
        return "classical"
    return m


def spectrum(t, c):
    tmax = t[-1]
    w = np.cos(np.pi * t / (2 * tmax)) ** 2
    y = (c * w)[None, :] * np.cos(np.outer(OMEGA, t))
    return np.trapezoid(y, t, axis=1)


def metrics(i_w, i_ref, peak_ref=None):
    cm = OMEGA * CM
    win = (cm >= 2800) & (cm <= 4400)
    iw = np.clip(i_w[win], 0.0, None)
    centre = float((cm[win] * iw).sum() / iw.sum())
    width = float(np.sqrt(((cm[win] - centre) ** 2 * iw).sum() / iw.sum()))
    k = int(np.argmax(i_w[win]))
    y0, y1, y2 = i_w[win][k - 1:k + 2]
    peak = float(cm[win][k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    out = {"centre_cm": centre, "width_cm": width, "peak_cm": peak}
    if peak_ref is not None:
        far = np.abs(cm - peak_ref) > 150
        out["spurious"] = float(np.trapezoid(np.abs(i_w - i_ref) * far, cm) / np.trapezoid(i_ref, cm))
    return out


def load(d):
    with open(os.path.join(d, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    c = np.genfromtxt(os.path.join(d, "correlations.csv"), delimiter=",", names=True)
    pos = cfg["system"]["positions_angstrom"]
    system = "s1" if all(abs(p[0] - p[1]) < 1e-12 and abs(p[1] - p[2]) < 1e-12 for p in pos) else "s2"
    return cfg, system, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    groups = defaultdict(list)
    for d in a.runs:
        cfg, system, c = load(d)
        key = (system, cfg["sampling"]["temperature_kelvin"], method_of(cfg), cfg["sampling"]["n_beads"],
               cfg["sampling"]["dt_fs"], cfg.get("dynamics", {}).get("cmd_omega_cm"))
        groups[key].append(c)
    results = []
    refs = {}
    for key, reps in sorted(groups.items(), key=lambda kv: str(kv[0])):
        system, tk, method, p, dt, omega = key
        t = reps[0]["t_fs"] * FS
        if (system, tk, len(t)) not in refs:
            r = exact.reference(system, tk, t)
            i_ref = spectrum(t, r["c_vv"])
            refs[system, tk, len(t)] = (r, i_ref, metrics(i_ref, i_ref))
        r, i_ref, m_ref = refs[system, tk, len(t)]
        per = [metrics(spectrum(t, c["c_vv_au"]), i_ref, m_ref["peak_cm"]) for c in reps]
        cvv = np.mean([c["c_vv_au"] for c in reps], axis=0)
        sem = np.sqrt(np.sum([c["c_vv_sem_au"] ** 2 for c in reps], axis=0)) / len(reps)
        m_all = metrics(spectrum(t, cvv), i_ref, m_ref["peak_cm"])
        early = t <= 500 * FS
        dev = np.abs(cvv - r["c_vv"])[early]
        row = {"system": system, "T": tk, "method": method, "P": p, "dt_fs": dt, "omega_cm": omega,
               "replicas": len(reps), **m_all,
               **{f"{k}_sigma": float(np.std([q[k] for q in per], ddof=1) / np.sqrt(len(per)))
                  if len(per) > 1 else None for k in m_all},
               "exact": m_ref,
               "cvv_err_rel": float(dev.max() / r["c_vv"][0]),
               "cvv_err_sigma": float((dev / np.maximum(sem[early], 1e-300)).max()),
               "cvv0_rel": float(cvv[0] / r["c_vv"][0] - 1),
               "cxx0_rel": float(np.mean([c["c_xx_bohr2"][0] for c in reps]) / r["c_xx"][0] - 1)}
        results.append(row)
        s = lambda k: f"{row[k]:8.1f}" + (f"±{row[k + '_sigma']:.1f}" if row.get(k + "_sigma") else "")
        print(f"{system} {tk:5.0f}K {method:9s} P={p:<3d} dt={dt} Ω={omega} n={len(reps)}: "
              f"centre {s('centre_cm')} (exact {m_ref['centre_cm']:.1f}), "
              f"peak {s('peak_cm')} (exact {m_ref['peak_cm']:.1f}), "
              f"width {s('width_cm')} (exact {m_ref['width_cm']:.1f}), spurious {row['spurious']:.3f}, "
              f"|ΔC_vv|/C(0) {row['cvv_err_rel']:.3f} ({row['cvv_err_sigma']:.1f}σ), "
              f"C_xx(0) {100 * row['cxx0_rel']:+.2f}%")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
