"""M11 tier 1: centroid velocity spectra of every method against exact (results/spectra.png).

    pixi run -e ref python studies/m11-dynamics-failure/plot_spectra.py

Production settings only (Δt = 0.1 fs; PA-CMD at 100 K: Δt = 0.1/3 fs); replicas averaged; same
transform as analyse.py (Hann window over the 2 ps lag range). Dotted lines: internal ring-polymer
mode frequencies ω_k = 2ω_P sin(kπ/P), k = 1 … 5.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse as A  # noqa: E402
import exact  # noqa: E402

OUT = os.path.join(HERE, "out")
RUNS = {  # (system, T) -> {method: glob patterns}
    (s, t): {
        "classical": [f"prod/{s}_T{t}_classical_r?"] + ([f"pilot/dyn/{s}_*classical*"] if t == 300 else []),
        "RPMD": [f"prod/{s}_T{t}_rpmd_r?"] + ([f"pilot/dyn/{s}_*_rpmd*"] if t == 300 else []),
        "TRPMD": [f"prod/{s}_T{t}_trpmd_r?"] + ([f"pilot/dyn/{s}_*trpmd*"] if t == 300 else []),
        "PA-CMD": [f"prod/{s}_T{t}_pa-cmd_r?"] + ([f"pilot/dyn/{s}_*pa-cmd*"] if t == 300 else [])
        if t == 300 else [f"prod/{s}_T{t}_pa-cmd_dt0.033_r?"],
    }
    for s in ("s1", "s2") for t in (300, 100)
}
P = {300: 64, 100: 192}
COLORS = {"classical": "#7f7f7f", "RPMD": "#1f77b4", "TRPMD": "#2ca02c", "PA-CMD": "#d62728"}


def main():
    cm = A.OMEGA * A.CM
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, ((s, t), methods) in zip(axes.ravel(), RUNS.items()):
        t_ref = None
        for name, pats in methods.items():
            dirs = sorted({d for p in pats for d in glob.glob(os.path.join(OUT, p))})
            reps = [np.genfromtxt(os.path.join(d, "correlations.csv"), delimiter=",", names=True)
                    for d in dirs]
            tt = reps[0]["t_fs"] * A.FS
            t_ref = tt
            i_w = A.spectrum(tt, np.mean([c["c_vv_au"] for c in reps], 0))
            ax.plot(cm, i_w / np.abs(i_w[cm >= 1500]).max(), color=COLORS[name], lw=1,
                    label=f"{name} (n = {len(reps)})")
        r = exact.reference(s, float(t), t_ref)
        i_ref = A.spectrum(t_ref, r["c_vv"])
        ax.plot(cm, i_ref / i_ref[cm >= 1500].max(), "k", lw=1.6, label="exact")
        w_p = P[t] * exact.K_B * t * A.CM
        for k in range(1, 6):
            ax.axvline(2 * w_p * np.sin(k * np.pi / P[t]), color="k", ls=":", lw=0.6)
        ax.set_title(f"{'S1 (1D)' if s == 's1' else 'S2 (3D radial)'}, {t} K, P = {P[t]}")
        ax.set_xlim(1500, 4500)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("ω (cm⁻¹)")
    for ax in axes[:, 0]:
        ax.set_ylabel("I(ω), normalised above 1500 cm⁻¹")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "spectra.png"), dpi=110)


if __name__ == "__main__":
    main()
