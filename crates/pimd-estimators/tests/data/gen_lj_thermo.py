"""LAMMPS NVE averages of the LJ liquid for the M3 thermodynamic comparison (MILESTONE_3; SPEC P-3,
I-4, E-8).

Run from the repository root:

    pixi run -e ref python crates/pimd-estimators/tests/data/gen_lj_thermo.py [--out DIR]

Writes lj_thermo.json next to this script (or into DIR). Input: the 256-atom snapshot (rho* = 0.8,
T* = 1, positions and velocities as exact doubles) from the committed
crates/pimd-potentials/tests/data/lj_snapshot.json (written by gen_lj_snapshot.py); regenerate that
file first if it changes.

Units: LAMMPS `units lj` with epsilon = sigma = m = 1 equals the engine's atomic units with
epsilon = 1 Eh, sigma = 1 bohr, m = 1 m_e (see gen_lj_snapshot.py); no conversions. Pressure from
`compute pressure` (thermo_press) is (2K + tr W)/(3V) exactly (dof*kB*T = sum m v^2), SPEC E-8.

Protocol (the engine test, lj_thermo.rs, runs the same): `pair_style lj/cut 2.5`,
`pair_modify shift yes`, skin 0.3 with `neigh_modify every 1 delay 0 check yes`, `fix nve`,
dt = 0.005, from the snapshot. N_EQUIL steps discarded, then N_PROD steps with the total potential
energy and the pressure sampled every SAMPLE steps (fix ave/time with Nrepeat = 1: instantaneous
values). Same initial microstate as the engine, hence the same total energy and the same
microcanonical ensemble; the trajectories decorrelate by chaos after a few thousand steps.

Statistics: mean, and the block-averaged standard error with blocks of BLOCK and 4*BLOCK samples
(std of block means / sqrt(n_blocks), trailing partial block dropped); sigma is the larger of the
two, and the generator asserts sigma_4block <= 1.5 sigma_block (blocks long enough to be
uncorrelated). This is the same procedure as `common::mean_and_sigma` in the engine test.

Smoke values: plain means of the same samples over the first N_SMOKE steps (no equilibration), for
the short default-suite version of the comparison. Over so few steps the engine and LAMMPS
trajectories still coincide (positions agree to ~1e-11 at step 400; the separation grows about
tenfold every 100 steps), so the smoke test compares these means at the trajectory level, not
statistically.

Reproducibility: serial LAMMPS is deterministic for a given binary (pinned by the pixi lock).
"""

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import lammps  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "crates/pimd-potentials/tests/data/lj_snapshot.json"
DT = 0.005
SKIN = 0.3
N_EQUIL = 2000
N_PROD = 200000
SAMPLE = 10
BLOCK = 100
N_SMOKE = 400


def conda_version(name):
    meta = Path(sys.prefix) / "conda-meta"
    for f in sorted(meta.glob(f"{name}-*.json")):
        info = json.loads(f.read_text())
        if info.get("name") == name:
            return f"{info['version']} ({info['build']})"
    return "unknown"


def block_sigma(x, length):
    nb = len(x) // length
    assert nb >= 20, nb
    means = x[: nb * length].reshape(nb, length).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nb))


def stats(x, length):
    s1, s4 = block_sigma(x, length), block_sigma(x, 4 * length)
    return {
        "mean": float(x.mean()),
        "sigma": max(s1, s4),
        "sigma_block": s1,
        "sigma_4block": s4,
        "n_samples": len(x),
        "block_samples": length,
    }, s1, s4


def run(snap, n_steps, sample, tmp):
    p = snap["parameters"]
    box = snap["box"]
    n = snap["n_atoms"]
    x = snap["positions"]
    v = snap["velocities"]
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lmp.commands_string(
        f"""
        units lj
        atom_style atomic
        atom_modify map array
        region box block 0 {box!r} 0 {box!r} 0 {box!r}
        create_box 1 box
        mass 1 {p["mass"]}
        pair_style lj/cut {p["cutoff"]}
        pair_coeff 1 1 {p["epsilon"]} {p["sigma"]} {p["cutoff"]}
        pair_modify shift yes
        neighbor {SKIN} bin
        neigh_modify every 1 delay 0 check yes
        """
    )
    lmp.create_atoms(n, list(range(1, n + 1)), [1] * n, x, v)
    out = Path(tmp) / f"series_{n_steps}.txt"
    lmp.commands_string(
        f"""
        fix 1 all nve
        timestep {DT}
        thermo 10000
        fix s all ave/time {sample} 1 {sample} c_thermo_pe c_thermo_press file {out} format " %.17g"
        run {n_steps}
        """
    )
    lmp.close()
    data = np.loadtxt(out, comments="#")
    assert np.array_equal(data[:, 0], np.arange(0, n_steps + 1, sample)), data[:3, 0]
    # Row 0 is step 0 (the snapshot itself); samples are taken after steps sample, 2*sample, ...
    return data[1:, 1], data[1:, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    snap = json.loads(SNAPSHOT.read_text())
    n = snap["n_atoms"]

    with tempfile.TemporaryDirectory() as tmp:
        pe, press = run(snap, N_EQUIL + N_PROD, SAMPLE, tmp)
        pe_s, press_s = run(snap, N_SMOKE, SAMPLE, tmp)
    skip = N_EQUIL // SAMPLE
    pe, press = pe[skip:], press[skip:]
    assert len(pe) == N_PROD // SAMPLE
    pe_stats, s1, s4 = stats(pe, BLOCK)
    assert s4 <= 1.5 * s1, ("pe", s1, s4)
    p_stats, s1, s4 = stats(press, BLOCK)
    assert s4 <= 1.5 * s1, ("pressure", s1, s4)
    smoke = {
        "n_steps": N_SMOKE,
        "n_samples": len(pe_s),
        "potential_energy_mean": float(pe_s.mean()),
        "pressure_mean": float(press_s.mean()),
    }

    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lmp_version = lmp.version()
    lmp.close()
    data = {
        "description": "LAMMPS NVE averages (total potential energy, pressure) of the 256-atom LJ "
        "liquid from the lj_snapshot.json state; lj units = engine atomic units",
        "generator": "gen_lj_thermo.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lammps": lmp_version,
            "lammps_conda": conda_version("lammps"),
        },
        "input": "crates/pimd-potentials/tests/data/lj_snapshot.json",
        "lammps_inputs": {
            "pair_style": f"lj/cut {snap['parameters']['cutoff']}",
            "pair_modify": "shift yes",
            "neighbor": f"{SKIN} bin",
            "neigh_modify": "every 1 delay 0 check yes",
            "integrator": f"fix nve; timestep {DT}",
            "sampling": f"fix ave/time {SAMPLE} 1 {SAMPLE} c_thermo_pe c_thermo_press",
        },
        "protocol": {
            "dt": DT,
            "n_atoms": n,
            "n_equil": N_EQUIL,
            "n_prod": N_PROD,
            "sample_every": SAMPLE,
        },
        "potential_energy": pe_stats,
        "pressure": p_stats,
        "smoke": smoke,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lj_thermo.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
