"""i-PI reference for the M4 liquid-neon PIMD validation (MILESTONE_4 "vs i-PI"; SPEC I-1, T-1,
T-2, E-1 ... E-4, E-7, P-3).

Run from the repository root:

    pixi run -e ref python crates/pimd-cli/tests/data/gen_ipi_neon.py [--out DIR]

Writes ipi_neon.json next to this script (or into DIR). The engine test (ipi_neon.rs, feature
long-validation) runs the same system with `pimd_cli::run` and compares <E_vir>, <E_prim> and g(r)
within 5 sigma_combined.

System (identical on both sides): N = 108 neon atoms, m = 20.1797 amu, cubic box 14.1 Angstrom
(rho* = 0.80), Lennard-Jones eps = 35.6 K, sigma = 2.749 Angstrom, r_c = 2.2 sigma, shifted
(LAMMPS `pair_modify shift yes` = SPEC P-3 shift), no tail correction; T = 30 K, P = 16 beads.
Start: the engine's pinned simple-cubic lattice (MILESTONE_4 "Initial state"), all beads at the
site; i-PI draws its own thermal momenta (seed SEED), so the trajectories differ from the engine's
and only the averages are compared.

Dynamics: i-PI `nvt` with `splitting="baoab"`: B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2) with the
exact free ring-polymer propagator in normal modes (i-PI qdt = dt/2, tdt = dt; engine/motion/
dynamics.py), the same splitting as SPEC I-1. Thermostat PILE-L with tau = 100 fs on the centroid
and pile_lambda = 1, i.e. gamma_k = 2 omega_k (i-PI tauk = 1/(2 lambda omega_k)), c = exp(-dt/tau),
noise at the bead temperature P T: SPEC T-1/T-2. Because the integrators are the same algorithm at
the same dt, their discretisation bias is common to both (see ipi_neon.rs for the dt/2 check).
dt = 2 fs. Forces: LAMMPS `fix ipi` (UNIX socket) with `units electron` (Hartree, bohr), so the
LJ parameters are passed in atomic units computed with the CODATA 2018 constants of SPEC U-2.
fix ipi converts with its own constants (Hartree<->LAMMPS energy, bohr<->LAMMPS length); for
`units electron` the factors are 1 to ~1e-7, far below the statistical errors.

Temperature, time step, tau and masses are given to i-PI in atomic units (k_B T in Hartree), so
i-PI's own kelvin/amu/fs conversion constants do not enter.

Protocol: N_EQUIL = 5000 steps discarded, then N_PROD = 200000 steps (400 ps). Properties every 10
steps (potential = bead-averaged V, kinetic_cv = SPEC E-2, kinetic_td = SPEC E-1; checked in
ipi/engine/properties.py), per atom: E_vir = (kinetic_cv + potential)/N, E_prim = (kinetic_td +
potential)/N. Bead positions every 100 steps; g(r) per frame by SPEC E-7 (minimum image, pairs i < j
within each bead, N_BINS bins on [0, r_max), C_b averaged over beads, g_b = 2 C_b V / (N (N - 1)
V_shell(b))).

Statistics: block averages. The potential energy of this small liquid decorrelates slowly (block
errors keep growing up to blocks of ~5-10 ps), hence the long run. Energies: blocks of 250 samples
(2500 steps, 5 ps) and 1000 samples (20 ps); sigma is the larger, and the generator asserts
sigma_4block <= 1.5 sigma_block. g(r): per bin, blocks of 25 frames (2500 steps) and 100 frames,
sigma the larger (no per-bin assertion: 70 noisy ratios). Same procedure as
`common::mean_and_sigma` in the engine test.

Reproducibility: i-PI's PRNG is seeded (SEED), there is one serial LAMMPS client, BLAS/OpenMP run
single-threaded, and the socket transfers doubles exactly. LAMMPS must not carry state from one
bead to the next: i-PI may send the beads in any order, and with `check yes` the neighbor list (and
so the force summation order) would depend on the previously evaluated bead, which made two
otherwise identical 205 000-step runs diverge (E_vir differed by 0.7 sigma). Hence
`neigh_modify every 1 delay 0 check no` (lists rebuilt from the current positions every call) and
`atom_modify sort 0 0.0` (no atom reordering). With these, a rerun on the same binaries reproduces
the trajectory bit for bit (checked by running the generator twice); verify_reference_data.py
compares at rtol 1e-9.
Run time: about 34 min (i-PI 3.3.1 + one serial LAMMPS client, 205 000 steps; printed at the end
of each run).
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# SPEC U-2 (CODATA 2018).
K_B = 3.1668115634556e-6  # Eh/K
AMU = 1822.888486209  # m_e
BOHR_A = 0.529177210903  # Angstrom per bohr
FS = 41.341373336561  # hbar/Eh per fs

N_ATOMS = 108
MASS_AMU = 20.1797
BOX_A = 14.1
EPS_K = 35.6
SIGMA_A = 2.749
CUTOFF_SIGMA = 2.2
SKIN_SIGMA = 0.3
T_K = 30.0
N_BEADS = 16
DT_FS = 2.0
TAU0_FS = 100.0
N_EQUIL = 5000
N_PROD = 200000
STRIDE = 10
TRAJ_STRIDE = 100
N_BINS = 70
RMAX_A = 7.0
BLOCK_STEPS = 2500
SEED = 31415


def conda_version(name):
    meta = Path(sys.prefix) / "conda-meta"
    for f in sorted(meta.glob(f"{name}-*.json")):
        info = json.loads(f.read_text())
        if info.get("name") == name:
            return f"{info['version']} ({info['build']})"
    return "unknown"


def lattice(n, length):
    """MILESTONE_4 initial lattice: n_side = ceil(n^(1/3)) (exact), sites ((i + 1/2) a), ix fastest."""
    side = 1
    while side**3 < n:
        side += 1
    a = length / side
    sites = []
    for iz in range(side):
        for iy in range(side):
            for ix in range(side):
                if len(sites) < n:
                    sites.append(((ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a))
    return sites


def block_sigma(x, length):
    nb = len(x) // length
    assert nb >= 20, nb
    means = x[: nb * length].reshape(nb, length, *x.shape[1:]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(nb)


def stats(x, length, check=True):
    s1, s4 = block_sigma(x, length), block_sigma(x, 4 * length)
    if check:
        assert s4 <= 1.5 * s1, (s1, s4)
    return {
        "mean": float(x.mean()),
        "sigma": float(max(s1, s4)),
        "sigma_block": float(s1),
        "sigma_4block": float(s4),
        "n_samples": len(x),
        "block_samples": length,
    }


def read_xyz_frames(path, n):
    lines = path.read_text().splitlines()
    per = n + 2
    assert len(lines) % per == 0, (path, len(lines))
    frames = []
    for f in range(len(lines) // per):
        block = lines[f * per + 2 : (f + 1) * per]
        frames.append([[float(v) for v in ln.split()[1:4]] for ln in block])
    return np.array(frames)


def rdf_frame(beads_xyz, length, n_bins, r_max):
    """g_b of one frame (all beads), SPEC E-7."""
    p, n, _ = beads_xyz.shape
    iu = np.triu_indices(n, 1)
    counts = np.zeros(n_bins)
    for x in beads_xyz:
        d = x[None, :, :] - x[:, None, :]
        d = d[iu]
        d -= length * np.floor(d / length + 0.5)
        r = np.sqrt((d * d).sum(axis=1))
        r = r[r < r_max]
        b = np.minimum((r * n_bins / r_max).astype(int), n_bins - 1)
        counts += np.bincount(b, minlength=n_bins)
    counts /= p
    edges = np.arange(n_bins + 1) * (r_max / n_bins)
    shell = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    return 2.0 * counts * length**3 / (n * (n - 1) * shell)


def write_inputs(tmp, address):
    length = BOX_A / BOHR_A
    sigma = SIGMA_A / BOHR_A
    eps = EPS_K * K_B
    rc = CUTOFF_SIGMA * sigma
    skin = SKIN_SIGMA * sigma
    mass = MASS_AMU * AMU
    sites = lattice(N_ATOMS, length)
    with open(tmp / "init.xyz", "w") as f:
        f.write(f"{N_ATOMS}\n# lattice, bohr\n")
        for s in sites:
            f.write(f"Ne {s[0]!r} {s[1]!r} {s[2]!r}\n")
    with open(tmp / "data.lmp", "w") as f:
        f.write(f"neon lattice\n\n{N_ATOMS} atoms\n1 atom types\n\n")
        for ax in "xyz":
            f.write(f"0.0 {length!r} {ax}lo {ax}hi\n")
        f.write(f"\nMasses\n\n1 {MASS_AMU!r}\n\nAtoms # atomic\n\n")
        for i, s in enumerate(sites):
            f.write(f"{i + 1} 1 {s[0]!r} {s[1]!r} {s[2]!r}\n")
    (tmp / "in.lmp").write_text(
        f"""units electron
atom_style atomic
atom_modify map yes sort 0 0.0
boundary p p p
read_data data.lmp
pair_style lj/cut {rc!r}
pair_coeff 1 1 {eps!r} {sigma!r}
pair_modify shift yes
neighbor {skin!r} bin
neigh_modify every 1 delay 0 check no
thermo_style one
thermo 1000000000
fix 1 all ipi {address} 31415 unix reset
run 2000000000
"""
    )
    masses = ", ".join([repr(mass)] * N_ATOMS)
    n_total = N_EQUIL + N_PROD
    (tmp / "input.xml").write_text(
        f"""<simulation verbosity='low' safe_stride='{n_total}'>
  <output prefix='neon'>
    <properties stride='{STRIDE}' filename='out' flush='1000'>
      [ step, potential{{atomic_unit}}, kinetic_cv{{atomic_unit}}, kinetic_td{{atomic_unit}} ]
    </properties>
    <trajectory filename='pos' stride='{TRAJ_STRIDE}' format='xyz' cell_units='atomic_unit' flush='1000'>
      positions{{atomic_unit}}
    </trajectory>
  </output>
  <total_steps> {n_total} </total_steps>
  <prng><seed> {SEED} </seed></prng>
  <ffsocket name='lammps' mode='unix' pbc='true'>
    <address> {address} </address>
    <latency> 1e-4 </latency>
  </ffsocket>
  <system>
    <initialize nbeads='{N_BEADS}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell mode='abc' units='atomic_unit'> [ {length!r}, {length!r}, {length!r} ] </cell>
      <masses mode='manual' units='atomic_unit'> [ {masses} ] </masses>
      <velocities mode='thermal' units='atomic_unit'> {T_K * K_B!r} </velocities>
    </initialize>
    <forces><force forcefield='lammps'/></forces>
    <ensemble><temperature units='atomic_unit'> {T_K * K_B!r} </temperature></ensemble>
    <motion mode='dynamics'>
      <dynamics mode='nvt' splitting='baoab'>
        <timestep units='atomic_unit'> {DT_FS * FS!r} </timestep>
        <thermostat mode='pile_l'>
          <tau units='atomic_unit'> {TAU0_FS * FS!r} </tau>
          <pile_lambda> 1.0 </pile_lambda>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
"""
    )
    return {
        "box_bohr": length,
        "sigma_bohr": sigma,
        "epsilon_eh": eps,
        "cutoff_bohr": rc,
        "skin_bohr": skin,
        "mass_me": mass,
        "beta": 1.0 / (T_K * K_B),
    }


def run_ipi(tmp):
    address = f"pimdneon{os.getpid()}"
    sock = Path(f"/tmp/ipi_{address}")
    if sock.exists():
        sock.unlink()
    derived = write_inputs(tmp, address)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ipi = subprocess.Popen(
        [shutil.which("i-pi"), "input.xml"], cwd=tmp, stdout=open(tmp / "ipi.log", "w"),
        stderr=subprocess.STDOUT, env=env,
    )
    for _ in range(600):
        if sock.exists():
            break
        assert ipi.poll() is None, (tmp / "ipi.log").read_text()
        time.sleep(0.1)
    lmp = subprocess.Popen(
        [shutil.which("lmp"), "-in", "in.lmp", "-log", "none", "-screen", "none", "-nocite"],
        cwd=tmp, env=env, stdout=open(tmp / "lmp.log", "w"), stderr=subprocess.STDOUT,
    )
    assert ipi.wait() == 0, (tmp / "ipi.log").read_text()[-3000:]
    lmp.wait()
    elapsed = time.time() - t0
    return derived, elapsed


def analyse(tmp, derived):
    n = N_ATOMS
    out = np.loadtxt(tmp / "neon.out", comments="#")
    step = out[:, 0].astype(int)
    prod = step > N_EQUIL
    assert prod.sum() == N_PROD // STRIDE, prod.sum()
    assert np.all((step[prod] - N_EQUIL) % STRIDE == 0)
    pot, kcv, ktd = out[prod, 1], out[prod, 2], out[prod, 3]
    e_vir = (kcv + pot) / n
    e_prim = (ktd + pot) / n
    block = BLOCK_STEPS // STRIDE

    length = derived["box_bohr"]
    r_max = RMAX_A / BOHR_A
    beads = [read_xyz_frames(tmp / f"neon.pos_{j:02d}.xyz", n) for j in range(N_BEADS)]
    nf = beads[0].shape[0]
    frame_steps = np.arange(nf) * TRAJ_STRIDE
    keep = np.nonzero(frame_steps > N_EQUIL)[0]
    assert len(keep) == N_PROD // TRAJ_STRIDE, (nf, len(keep))
    g = np.array([rdf_frame(np.array([b[f] for b in beads]), length, N_BINS, r_max) for f in keep])
    gblock = BLOCK_STEPS // TRAJ_STRIDE
    s1, s4 = block_sigma(g, gblock), block_sigma(g, 4 * gblock)
    edges = np.arange(N_BINS + 1) * (r_max / N_BINS)
    return {
        "e_vir": stats(e_vir, block),
        "e_prim": stats(e_prim, block),
        "v": stats(pot / n, block),
        "rdf": {
            "r_bohr": (0.5 * (edges[1:] + edges[:-1])).tolist(),
            "g": g.mean(axis=0).tolist(),
            "sigma": np.maximum(s1, s4).tolist(),
            "n_frames": len(keep),
            "block_frames": gblock,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        derived, elapsed = run_ipi(tmp)
        results = analyse(tmp, derived)
    import ipi

    data = {
        "description": "i-PI + LAMMPS fix ipi PIMD of liquid neon (LJ), per-atom E_vir, E_prim "
        "(Eh/atom), V and g(r) with block-averaged errors",
        "generator": "gen_ipi_neon.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
            "lammps_conda": conda_version("lammps"),
        },
        "system": {
            "n_atoms": N_ATOMS,
            "mass_amu": MASS_AMU,
            "box_angstrom": BOX_A,
            "epsilon_kelvin": EPS_K,
            "sigma_angstrom": SIGMA_A,
            "cutoff_sigma": CUTOFF_SIGMA,
            "skin_sigma": SKIN_SIGMA,
            "shift": True,
            "temperature_kelvin": T_K,
            "n_beads": N_BEADS,
            "dt_fs": DT_FS,
            "tau0_fs": TAU0_FS,
            "rdf_bins": N_BINS,
            "rdf_rmax_angstrom": RMAX_A,
        },
        "derived_atomic_units": derived,
        "protocol": {
            "n_equil": N_EQUIL,
            "n_prod": N_PROD,
            "stride": STRIDE,
            "traj_stride": TRAJ_STRIDE,
            "block_steps": BLOCK_STEPS,
            "seed": SEED,
            "splitting": "baoab",
            "thermostat": "pile_l, pile_lambda = 1",
        },
        **results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_neon.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; i-PI + LAMMPS run time {elapsed:.0f} s")


if __name__ == "__main__":
    main()
