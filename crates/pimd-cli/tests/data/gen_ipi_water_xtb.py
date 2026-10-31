"""i-PI reference for the M5 water / GFN2-xTB validation (MILESTONE_5 "Long validation: Water on
GFN2-xTB vs i-PI"; SPEC I-1, T-1, T-2, E-1 ... E-4, K-1 ... K-6).

Run from the repository root:

    pixi run -e ref python crates/pimd-cli/tests/data/gen_ipi_water_xtb.py [--out DIR]

Writes ipi_water_xtb.json next to this script (or into DIR). The engine test long_water_xtb.rs
(feature socket-validation, run only by CHECK_LONG=1 CHECK_IPI=1 scripts/check.sh) runs the same
system with `pimd_cli::run` through the socket potential, driving the same force client, and
compares the averages within 5 sigma_combined.

System (identical on both sides): one gas-phase water molecule, O H H with masses 15.999, 1.008,
1.008 amu, starting geometry of the brief (Angstrom: O (0, 0, 0.1173), H (0, +-0.7572, -0.4692)),
all beads at the start geometry; T = 300 K, P = 16 beads. Forces: GFN2-xTB (tblite) single points
from crates/pimd-socket/tests/clients/ipi_client.py, backend xtb, N_CLIENTS clients, each
single-threaded and stateless. i-PI must send a cell; the clients run with --pbc none (open
boundaries), and i-PI's ffsocket has pbc='false' (positions are sent unwrapped). The engine sends
the gas-phase zero cell (SPEC K-3) to the same clients.

Dynamics: i-PI `nvt` with `splitting="baoab"` (B A O A B with the exact free ring-polymer
propagator, the same splitting as SPEC I-1), PILE-L with centroid tau = TAU0_FS and
pile_lambda = 1 (gamma_k = 2 omega_k, SPEC T-2), fixcom = False (the engine does not remove the
centre-of-mass motion; with fixcom the internal ensemble would be the same, but the noise on the
centroid would differ). Because both codes integrate with the same splitting at the same dt, their
discretisation bias is common (see long_water_xtb.rs for the dt/2 check of the size of the bias).
Temperature, time step, tau and masses are given to i-PI in atomic units (k_B T in Hartree, masses
in m_e), computed with the CODATA 2018 constants of SPEC U-2, so i-PI's own conversion constants do
not enter. i-PI draws its own thermal momenta (seed SEED); trajectories differ from the engine's
and only averages are compared.

Observables, per molecule (not per atom), Hartree: from i-PI properties every STRIDE steps,
E_vir = kinetic_cv + potential (SPEC E-2 + E-3; ipi/engine/properties.py get_kincv: 3N/(2 beta)
- (1/2P) sum (q - q_c) . f), E_prim = kinetic_td + potential (SPEC E-1 + E-3), V = potential
(bead average). From bead positions every TRAJ_STRIDE steps: the bead-averaged O-H distance
r_OH (mean over the 16 beads and both bonds, bohr), the bead-averaged H-O-H angle theta (degrees),
and the r_OH histogram (density over HIST_BINS bins on [HIST_LO, HIST_HI) bohr, both bonds of every
bead counted; values outside the range are not counted but the density is normalised by the
total number of distances, so it integrates to the fraction inside).

Protocol: N_EQUIL steps discarded, N_PROD production steps. Statistics: block averages, energies
in blocks of BLOCK_STEPS steps and 4 BLOCK_STEPS (sigma the larger; asserts sigma_4 <= 1.5 sigma_1),
geometry likewise from the frames; histogram per bin with the same blocks (no assertion per bin).
Same procedure as `common::mean_and_sigma` in the engine test.

Reproducibility: i-PI's PRNG is seeded; the clients are deterministic and keep no state between
calls (a fresh tblite Calculator per call, one OpenMP/BLAS thread), so the results do not depend
on which client evaluates which bead or when; the socket transfers doubles exactly. Two runs of
this script give byte-identical JSON (checked: two runs at the same time, compared with cmp).

Run time: i-PI's own overhead dominates (about 8 ms per step with four clients, 0.4 ms per xTB
single point), so about 55 min for the 404 000 steps on this machine with its usual background
load; 83 min (4 995 s) when measured with three other runs of this validation sharing the cores.
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

# SPEC U-2 (CODATA 2018).
K_B = 3.1668115634556e-6  # Eh/K
AMU = 1822.888486209  # m_e
BOHR_A = 0.529177210903  # Angstrom per bohr
FS = 41.341373336561  # hbar/Eh per fs

SYMBOLS = ["O", "H", "H"]
MASSES_AMU = [15.999, 1.008, 1.008]
POSITIONS_A = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
T_K = 300.0
N_BEADS = 16
DT_FS = 0.5
TAU0_FS = 20.0
N_EQUIL = 4000
N_PROD = 400000
STRIDE = 10
TRAJ_STRIDE = 20
BLOCK_STEPS = 4000
HIST_LO, HIST_HI, HIST_BINS = 1.4, 2.4, 25
N_CLIENTS = 4
SEED = 27182
CELL_BOHR = 40.0  # i-PI needs a cell; unused (clients with --pbc none, ffsocket pbc='false')

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent.parent.parent / "pimd-socket" / "tests" / "clients" / "ipi_client.py"


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
    means = x[: nb * length].reshape(nb, length, *x.shape[1:]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(nb)


def stats(x, length):
    s1, s4 = block_sigma(x, length), block_sigma(x, 4 * length)
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


def geometry(beads_xyz):
    """Per-frame bead-averaged r_OH (bohr), theta_HOH (degrees) and the 2P O-H distances.

    beads_xyz: [frames, P, 3 atoms, 3] in bohr, atoms O, H, H.
    """
    d1 = beads_xyz[:, :, 1] - beads_xyz[:, :, 0]
    d2 = beads_xyz[:, :, 2] - beads_xyz[:, :, 0]
    r1 = np.sqrt((d1 * d1).sum(axis=-1))
    r2 = np.sqrt((d2 * d2).sum(axis=-1))
    cos = (d1 * d2).sum(axis=-1) / (r1 * r2)
    theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    r = np.concatenate([r1, r2], axis=1)  # [frames, 2P]
    return r.mean(axis=1), theta.mean(axis=1), r


def histogram(r):
    """Per-frame density of the O-H distances r [frames, k] over HIST_BINS bins."""
    width = (HIST_HI - HIST_LO) / HIST_BINS
    idx = np.floor((r - HIST_LO) / width).astype(int)
    out = np.zeros((r.shape[0], HIST_BINS))
    for f in range(r.shape[0]):
        ok = (idx[f] >= 0) & (idx[f] < HIST_BINS)
        out[f] = np.bincount(idx[f][ok], minlength=HIST_BINS)
    return out / (r.shape[1] * width)


def derived():
    return {
        "masses_me": [m * AMU for m in MASSES_AMU],
        "positions_bohr": [[x / BOHR_A for x in p] for p in POSITIONS_A],
        "beta": 1.0 / (T_K * K_B),
        "dt": DT_FS * FS,
        "tau0": TAU0_FS * FS,
    }


def write_inputs(tmp, address, n_steps):
    d = derived()
    with open(tmp / "init.xyz", "w") as f:
        f.write(f"3\n# start geometry, bohr\n")
        for s, p in zip(SYMBOLS, d["positions_bohr"]):
            f.write(f"{s} {p[0]!r} {p[1]!r} {p[2]!r}\n")
    masses = ", ".join(repr(m) for m in d["masses_me"])
    kt = T_K * K_B
    (tmp / "input.xml").write_text(
        f"""<simulation verbosity='low' safe_stride='{n_steps}'>
  <output prefix='water'>
    <properties stride='{STRIDE}' filename='out' flush='1000'>
      [ step, potential{{atomic_unit}}, kinetic_cv{{atomic_unit}}, kinetic_td{{atomic_unit}} ]
    </properties>
    <trajectory filename='pos' stride='{TRAJ_STRIDE}' format='xyz' cell_units='atomic_unit' flush='1000'>
      positions{{atomic_unit}}
    </trajectory>
  </output>
  <total_steps> {n_steps} </total_steps>
  <prng><seed> {SEED} </seed></prng>
  <ffsocket name='xtb' mode='unix' pbc='false'>
    <address> {address} </address>
    <latency> 1e-4 </latency>
  </ffsocket>
  <system>
    <initialize nbeads='{N_BEADS}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell mode='abc' units='atomic_unit'> [ {CELL_BOHR!r}, {CELL_BOHR!r}, {CELL_BOHR!r} ] </cell>
      <masses mode='manual' units='atomic_unit'> [ {masses} ] </masses>
      <velocities mode='thermal' units='atomic_unit'> {kt!r} </velocities>
    </initialize>
    <forces><force forcefield='xtb'/></forces>
    <ensemble><temperature units='atomic_unit'> {kt!r} </temperature></ensemble>
    <motion mode='dynamics'>
      <fixcom> False </fixcom>
      <dynamics mode='nvt' splitting='baoab'>
        <timestep units='atomic_unit'> {d["dt"]!r} </timestep>
        <thermostat mode='pile_l'>
          <tau units='atomic_unit'> {d["tau0"]!r} </tau>
          <pile_lambda> 1.0 </pile_lambda>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
"""
    )


def run_ipi(tmp, n_steps):
    address = f"pimdwater{os.getpid()}"
    sock = Path(f"/tmp/ipi_{address}")
    if sock.exists():
        sock.unlink()
    write_inputs(tmp, address, n_steps)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ipi = subprocess.Popen(
        [shutil.which("i-pi"), "input.xml"], cwd=tmp, stdout=open(tmp / "ipi.log", "w"),
        stderr=subprocess.STDOUT, env=env,
    )
    clients = []
    try:
        for _ in range(600):
            if sock.exists():
                break
            assert ipi.poll() is None, (tmp / "ipi.log").read_text()
            time.sleep(0.1)
        for k in range(N_CLIENTS):
            clients.append(
                subprocess.Popen(
                    [sys.executable, str(CLIENT), "--address", f"unix:{sock}", "--pbc", "none",
                     "xtb", "--symbols", *SYMBOLS],
                    cwd=tmp, env=env, stdout=open(tmp / f"client{k}.log", "w"),
                    stderr=subprocess.STDOUT,
                )
            )
        assert ipi.wait() == 0, (tmp / "ipi.log").read_text()[-3000:]
    finally:
        for c in clients:
            try:
                c.wait(timeout=30)
            except subprocess.TimeoutExpired:
                c.kill()
        if ipi.poll() is None:
            ipi.kill()
    return time.time() - t0


def analyse(tmp, n_equil=N_EQUIL, n_prod=N_PROD):
    out = np.loadtxt(tmp / "water.out", comments="#")
    step = out[:, 0].astype(int)
    prod = step > n_equil
    assert prod.sum() == n_prod // STRIDE, prod.sum()
    assert np.all((step[prod] - n_equil) % STRIDE == 0)
    pot, kcv, ktd = out[prod, 1], out[prod, 2], out[prod, 3]
    block = BLOCK_STEPS // STRIDE

    beads = np.stack(
        [read_xyz_frames(tmp / f"water.pos_{j:02d}.xyz", 3) for j in range(N_BEADS)], axis=1
    )
    frame_steps = np.arange(beads.shape[0]) * TRAJ_STRIDE
    keep = frame_steps > n_equil
    assert keep.sum() == n_prod // TRAJ_STRIDE, keep.sum()
    r_mean, theta, r_all = geometry(beads[keep])
    hist = histogram(r_all)
    tblock = BLOCK_STEPS // TRAJ_STRIDE
    s1, s4 = block_sigma(hist, tblock), block_sigma(hist, 4 * tblock)
    edges = HIST_LO + np.arange(HIST_BINS + 1) * (HIST_HI - HIST_LO) / HIST_BINS
    return {
        "e_vir": stats(kcv + pot, block),
        "e_prim": stats(ktd + pot, block),
        "v": stats(pot, block),
        "r_oh": stats(r_mean, tblock),
        "theta_hoh_deg": stats(theta, tblock),
        "r_oh_hist": {
            "edges_bohr": edges.tolist(),
            "density": hist.mean(axis=0).tolist(),
            "sigma": np.maximum(s1, s4).tolist(),
            "n_frames": int(keep.sum()),
            "block_frames": tblock,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        elapsed = run_ipi(tmp, N_EQUIL + N_PROD)
        results = analyse(tmp)
    import ipi

    data = {
        "description": "i-PI PIMD of gas-phase water on GFN2-xTB (tblite, ipi_client.py xtb): "
        "E_vir, E_prim, V per molecule (Eh), bead-averaged r_OH (bohr) and theta_HOH (deg), "
        "r_OH histogram, with block-averaged errors",
        "generator": "gen_ipi_water_xtb.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
            "tblite_conda": conda_version("tblite-python"),
        },
        "system": {
            "symbols": SYMBOLS,
            "masses_amu": MASSES_AMU,
            "positions_angstrom": POSITIONS_A,
            "temperature_kelvin": T_K,
            "n_beads": N_BEADS,
            "dt_fs": DT_FS,
            "tau0_fs": TAU0_FS,
        },
        "derived_atomic_units": derived(),
        "protocol": {
            "n_equil": N_EQUIL,
            "n_prod": N_PROD,
            "stride": STRIDE,
            "traj_stride": TRAJ_STRIDE,
            "block_steps": BLOCK_STEPS,
            "hist_lo_bohr": HIST_LO,
            "hist_hi_bohr": HIST_HI,
            "hist_bins": HIST_BINS,
            "n_clients": N_CLIENTS,
            "seed": SEED,
            "splitting": "baoab",
            "thermostat": "pile_l, pile_lambda = 1, fixcom = False",
        },
        **results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_water_xtb.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; i-PI + {N_CLIENTS} xtb clients run time {elapsed:.0f} s")


if __name__ == "__main__":
    main()
