"""LAMMPS reference data for the M3 Lennard-Jones tests (MILESTONE_3; SPEC P-3, P-5, I-4, E-7, E-8).

Run from the repository root:

    pixi run -e ref python crates/pimd-potentials/tests/data/gen_lj_snapshot.py [--out DIR]

Writes lj_snapshot.json next to this script (or into DIR). Consumers: pimd-potentials (single point:
forces, energy, virial), pimd-integrators (20-step NVE trajectory), pimd-estimators (pressure, RDF,
and the snapshot as the start of the thermodynamic comparison, gen_lj_thermo.py).

Units. LAMMPS `units lj` with epsilon = sigma = m = 1 maps one to one onto the engine's atomic units
(SPEC U-1) if the engine is given epsilon = 1 Eh, sigma = 1 bohr and m = 1 m_e: length sigma = 1 bohr,
energy epsilon = 1 Eh, mass 1 m_e, time sigma*sqrt(m/epsilon) = bohr*sqrt(m_e/Eh) = hbar/Eh (the atomic
unit of time), force epsilon/sigma = Eh/bohr, pressure epsilon/sigma^3 = Eh/bohr^3, and LAMMPS
`nktv2p` = `ftm2v` = 1 in lj units. Momenta are m*v = v. No conversion factors are needed anywhere.

Conventions checked here (documented, not worked around):
- `pair_style lj/cut` energy 4 eps [(s/r)^12 - (s/r)^6] for r^2 < rc^2, forces the exact derivative,
  `pair_modify shift yes` subtracts V(rc) from the energy only, no tail correction unless
  `pair_modify tail yes` (not used): SPEC P-3. A numpy brute-force sum below confirms LAMMPS energies
  and forces to 1e-11 relative for both shift settings.
- `compute pressure` scalar is (dof*kB*T + W_xx + W_yy + W_zz) / (3V), and dof*kB*T = sum m v^2 = 2K
  whatever the dof count (3N - 3 here), so it is SPEC E-8, p = (2K + tr W)/(3V). With the `virial`
  keyword and temperature NULL it returns W_ab / V in the order xx, yy, zz, xy, xz, yz, with
  W = sum_{i<j} r_ij (x) f_ij (r_ij = r_i - r_j, f_ij the force on i from j), the SPEC E-8 / P-5 sign.
  The numpy check below confirms the virial tensor.
- `compute rdf Nbin cutoff Rc`: bins [b, b+1) * Rc/Nbin, column 1 the bin centre (b + 1/2) Rc/Nbin,
  column 2 g = hist / (normfac * icount * (4 pi / 3V)(r_{b+1}^3 - r_b^3)), where for a single type
  hist counts every pair twice and normfac * icount = (N - 1) N, i.e. g_b = 2 C_b V / (N (N-1) Vshell),
  SPEC E-7 with P = 1 (checked below against a numpy histogram). No convention difference.
- `fix nve` is velocity Verlet (v += dt/2 f/m, x += dt v, forces, v += dt/2 f/m): SPEC I-4.

The snapshot: fcc lattice at rho* = 0.8 (256 atoms, cubic box L = 4 (4/0.8)^(1/3)), velocities drawn
at T* = 3 (LAMMPS `velocity create`, seed below), melted and equilibrated with a Nose-Hoover NVT run
at T* = 1 (50 tau, dt 0.005), then velocities rescaled to T* = 1 (dof 3N - 3). The positions are
wrapped into [0, L) and stored; every later LAMMPS run starts from these stored doubles, so the
engine and LAMMPS see bit-identical input. Reproducibility: serial LAMMPS is deterministic for a
given binary (pinned by the pixi lock), so regeneration reproduces the JSON.

Trajectory: 20 steps of `fix nve`, dt = 0.005, from the snapshot (shift yes; the shift does not
change forces). Unwrapped positions (`xu`) and velocities after steps 1, 2, 5, 10, 20, and PE at
every step.
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
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY  # noqa: E402

RHO = 0.8
CELLS = 4  # 4^3 fcc cells x 4 atoms = 256
EPSILON = 1.0
SIGMA = 1.0
MASS = 1.0
CUTOFF = 2.5
SKIN = 0.3
T_MELT = 3.0
T_TARGET = 1.0
VELOCITY_SEED = 4928459
N_EQUIL = 10000
DT = 0.005
N_TRAJ = 20
TRAJ_SAVE = [1, 2, 5, 10, 20]
RDF_BINS = 68
RDF_RMAX = 3.4


def new_lammps():
    return lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def conda_version(name):
    meta = Path(sys.prefix) / "conda-meta"
    for f in sorted(meta.glob(f"{name}-*.json")):
        info = json.loads(f.read_text())
        if info.get("name") == name:
            return f"{info['version']} ({info['build']})"
    return "unknown"


def equilibrate():
    """Session 1: melt the fcc lattice and equilibrate at T* = 1; return L, x (wrapped), v."""
    lmp = new_lammps()
    lmp.commands_string(
        f"""
        units lj
        atom_style atomic
        atom_modify map array
        lattice fcc {RHO}
        region box block 0 {CELLS} 0 {CELLS} 0 {CELLS}
        create_box 1 box
        create_atoms 1 box
        mass 1 {MASS}
        pair_style lj/cut {CUTOFF}
        pair_coeff 1 1 {EPSILON} {SIGMA} {CUTOFF}
        pair_modify shift yes
        neighbor {SKIN} bin
        neigh_modify every 1 delay 0 check yes
        velocity all create {T_MELT} {VELOCITY_SEED} mom yes rot no dist gaussian
        fix 1 all nvt temp {T_TARGET} {T_TARGET} 0.5
        timestep {DT}
        run {N_EQUIL}
        unfix 1
        velocity all scale {T_TARGET}
        """
    )
    lo, hi = lmp.extract_box()[:2]
    assert lo == [0.0, 0.0, 0.0], lo
    assert hi[0] == hi[1] == hi[2], hi
    box = hi[0]
    n = lmp.get_natoms()
    x = np.array(lmp.gather_atoms("x", 1, 3)[:], dtype=float).reshape(n, 3)
    v = np.array(lmp.gather_atoms("v", 1, 3)[:], dtype=float).reshape(n, 3)
    lmp.close()
    x = np.mod(x, box)
    x[x >= box] -= box
    assert np.all((x >= 0.0) & (x < box))
    return box, x, v


def setup(lmp, box, x, v, shift):
    """Session 2+: the system from the stored doubles, `pair_modify shift` yes or no."""
    lmp.commands_string(
        f"""
        units lj
        atom_style atomic
        atom_modify map array
        region box block 0 {box!r} 0 {box!r} 0 {box!r}
        create_box 1 box
        mass 1 {MASS}
        pair_style lj/cut {CUTOFF}
        pair_coeff 1 1 {EPSILON} {SIGMA} {CUTOFF}
        pair_modify shift {"yes" if shift else "no"}
        neighbor {SKIN} bin
        neigh_modify every 1 delay 0 check yes
        """
    )
    n = len(x)
    lmp.create_atoms(n, list(range(1, n + 1)), [1] * n, x.ravel().tolist(), v.ravel().tolist())
    assert lmp.get_natoms() == n
    lo, hi = lmp.extract_box()[:2]
    assert hi == [box, box, box], (hi, box)
    xs = np.array(lmp.gather_atoms("x", 1, 3)[:], dtype=float).reshape(n, 3)
    assert np.array_equal(xs, x), "LAMMPS did not keep the stored positions bit for bit"


def single_point(box, x, v, shift):
    lmp = new_lammps()
    setup(lmp, box, x, v, shift)
    lmp.commands_string(
        """
        compute ke all ke
        compute pv all pressure NULL virial
        thermo_style custom step pe ke press c_pv[1] c_pv[2] c_pv[3] c_pv[4] c_pv[5] c_pv[6]
        run 0
        """
    )
    n = len(x)
    f = np.array(lmp.gather_atoms("f", 1, 3)[:], dtype=float).reshape(n, 3)
    pe = lmp.extract_compute("thermo_pe", 0, 0)
    ke = lmp.extract_compute("ke", 0, 0)
    press = lmp.extract_compute("thermo_press", 0, 0)
    pv = [lmp.extract_compute("pv", 0, 1)[k] for k in range(6)]
    lmp.close()
    vol = box**3
    xx, yy, zz, xy, xz, yz = (vol * c for c in pv)
    virial = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return {
        "forces": f.ravel().tolist(),
        "potential_energy": pe,
        "kinetic_energy": ke,
        "pressure": press,
        "virial": virial,
    }


def numpy_check(box, x, v, sp, shift):
    """Brute-force minimum-image LJ (SPEC P-3, P-5, E-8) against the LAMMPS single point."""
    n = len(x)
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < CUTOFF**2
    sr6 = np.where(inside, (SIGMA**2 / r2) ** 3, 0.0)
    vshift = 4 * EPSILON * ((SIGMA / CUTOFF) ** 12 - (SIGMA / CUTOFF) ** 6) if shift else 0.0
    e_pair = np.where(inside, 4 * EPSILON * (sr6 * sr6 - sr6) - vshift, 0.0)
    fr_over_r = np.where(inside, 24 * EPSILON * (2 * sr6 * sr6 - sr6) / r2, 0.0)
    fij = fr_over_r[:, :, None] * d
    f = fij.sum(axis=1)
    e = 0.5 * e_pair.sum()
    w = 0.5 * np.einsum("ija,ijb->ab", d, fij)
    ke = 0.5 * MASS * np.sum(v * v)
    p = (2 * ke + np.trace(w)) / (3 * box**3)
    assert abs(e - sp["potential_energy"]) <= 1e-11 * abs(e), (e, sp["potential_energy"])
    fl = np.array(sp["forces"]).reshape(n, 3)
    assert np.max(np.abs(f - fl)) <= 1e-11 * np.max(np.abs(f)), np.max(np.abs(f - fl))
    assert np.max(np.abs(w - np.array(sp["virial"]))) <= 1e-11 * np.max(np.abs(w))
    assert abs(ke - sp["kinetic_energy"]) <= 1e-12 * ke
    assert abs(p - sp["pressure"]) <= 1e-11 * abs(p), (p, sp["pressure"])
    r = np.sqrt(r2[np.triu_indices(n, 1)])
    return r


def trajectory(box, x, v):
    lmp = new_lammps()
    setup(lmp, box, x, v, shift=True)
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "traj.txt"
        lmp.commands_string(
            f"""
            fix 1 all nve
            timestep {DT}
            compute ke all ke
            thermo_style custom step pe ke etotal
            dump d all custom 1 {dump} id xu yu zu vx vy vz
            dump_modify d sort id format float %.17g
            fix pe all ave/time 1 1 1 c_thermo_pe c_ke file {Path(tmp) / "pe.txt"} format " %.17g"
            run {N_TRAJ}
            """
        )
        lmp.close()
        frames = {}
        lines = dump.read_text().splitlines()
        i = 0
        while i < len(lines):
            assert lines[i] == "ITEM: TIMESTEP"
            step = int(lines[i + 1])
            n = int(lines[i + 3])
            body = lines[i + 9 : i + 9 + n]
            rows = np.array([[float(t) for t in ln.split()] for ln in body])
            assert np.array_equal(rows[:, 0], np.arange(1, n + 1))
            frames[step] = rows[:, 1:]
            i += 9 + n
        series = np.loadtxt(Path(tmp) / "pe.txt", comments="#")
    assert np.array_equal(frames[0][:, :3], x) and np.array_equal(frames[0][:, 3:], v)
    assert np.array_equal(series[:, 0], np.arange(0, N_TRAJ + 1))
    saved = [
        {
            "step": s,
            "positions": frames[s][:, :3].ravel().tolist(),
            "velocities": frames[s][:, 3:].ravel().tolist(),
        }
        for s in TRAJ_SAVE
    ]
    return {
        "dt": DT,
        "n_steps": N_TRAJ,
        "potential_energy": series[:, 1].tolist(),
        "kinetic_energy": series[:, 2].tolist(),
        "frames": saved,
    }


def rdf(box, x, v, r):
    lmp = new_lammps()
    setup(lmp, box, x, v, shift=True)
    lmp.commands_string(
        f"""
        comm_modify cutoff {RDF_RMAX + SKIN + 0.1}
        compute rdf all rdf {RDF_BINS} cutoff {RDF_RMAX}
        fix r all ave/time 1 1 1 c_rdf[*] mode vector
        run 0
        """
    )
    arr = lmp.numpy.extract_compute("rdf", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
    arr = np.array(arr, dtype=float).copy()
    lmp.close()
    assert arr.shape == (RDF_BINS, 3), arr.shape
    # numpy E-7 with P = 1 on the same pair distances.
    n = len(x)
    delr = RDF_RMAX / RDF_BINS
    edges = np.arange(RDF_BINS + 1) * delr
    rin = r[r < RDF_RMAX]
    # No pair distance within 1e-9 of a bin edge, so the bin assignment is unambiguous.
    gap = np.min(np.abs(rin[:, None] - edges[None, :]))
    assert gap > 1e-9, gap
    counts = np.array([np.sum((rin >= edges[b]) & (rin < edges[b + 1])) for b in range(RDF_BINS)])
    vshell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = 2 * counts * box**3 / (n * (n - 1) * vshell)
    assert np.allclose(arr[:, 1], g, rtol=1e-12, atol=0), np.max(np.abs(arr[:, 1] - g))
    assert np.allclose(arr[:, 0], (np.arange(RDF_BINS) + 0.5) * delr, rtol=1e-14)
    return {
        "n_bins": RDF_BINS,
        "r_max": RDF_RMAX,
        "bin_centers": arr[:, 0].tolist(),
        "g": arr[:, 1].tolist(),
        "pair_counts": counts.tolist(),
        "min_distance_to_bin_edge": float(gap),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    box, x, v = equilibrate()
    points = {}
    r = None
    for shift in (True, False):
        sp = single_point(box, x, v, shift)
        r = numpy_check(box, x, v, sp, shift)
        points["shift_yes" if shift else "shift_no"] = sp
    # No pair distance within 1e-9 of the cutoff, so r < rc is unambiguous.
    cut_gap = float(np.min(np.abs(r - CUTOFF)))
    assert cut_gap > 1e-9, cut_gap
    traj = trajectory(box, x, v)
    g = rdf(box, x, v, r)

    lmp = new_lammps()
    lmp_version = lmp.version()
    lmp.close()
    data = {
        "description": "256-atom LJ liquid snapshot (rho* = 0.8, T* = 1) and LAMMPS lj/cut "
        "single points, 20-step NVE trajectory and RDF; LAMMPS lj units = engine atomic units "
        "with epsilon = sigma = m = 1",
        "generator": "gen_lj_snapshot.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lammps": lmp_version,
            "lammps_conda": conda_version("lammps"),
        },
        "lammps_inputs": {
            "units": "lj",
            "pair_style": f"lj/cut {CUTOFF}",
            "pair_coeff": f"1 1 {EPSILON} {SIGMA} {CUTOFF}",
            "neighbor": f"{SKIN} bin",
            "neigh_modify": "every 1 delay 0 check yes",
            "melt": f"lattice fcc {RHO}; region 0 {CELLS} (x3); velocity create {T_MELT} "
            f"{VELOCITY_SEED} mom yes rot no dist gaussian; fix nvt temp {T_TARGET} {T_TARGET} 0.5; "
            f"timestep {DT}; run {N_EQUIL}; velocity all scale {T_TARGET}",
            "single_point": "run 0; compute pressure NULL virial; thermo_press",
            "trajectory": f"fix nve; timestep {DT}; run {N_TRAJ}; dump xu yu zu vx vy vz",
            "rdf": f"compute rdf {RDF_BINS} cutoff {RDF_RMAX}; comm_modify cutoff "
            f"{RDF_RMAX + SKIN + 0.1}; run 0",
        },
        "parameters": {
            "epsilon": EPSILON,
            "sigma": SIGMA,
            "mass": MASS,
            "cutoff": CUTOFF,
            "skin": SKIN,
            "density": RHO,
            "temperature": T_TARGET,
        },
        "n_atoms": len(x),
        "box": box,
        "positions": x.ravel().tolist(),
        "velocities": v.ravel().tolist(),
        "min_distance_to_cutoff": cut_gap,
        "shift_yes": points["shift_yes"],
        "shift_no": points["shift_no"],
        "trajectory": traj,
        "rdf": g,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lj_snapshot.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
