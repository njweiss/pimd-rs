"""i-PI reference for the M8a comparison of periodic MACE-OFF23 water with i-PI (MILESTONE_8 "M8a:
Required tests" 8; SPEC P-6, I-1, T-1, T-2, E-2, E-3, C-2, K-1 ... K-6).

    PIMD_MACE_MODEL=ff_models/mace_off23/MACE-OFF23_small.model \
        pixi run -e ref python crates/pimd-cli/tests/data/gen_ipi_water_mace.py [--out DIR]

Writes ipi_water_mace.json next to this script (or into DIR). Slow (about 4 h on the RTX 4060):
registered as slow in validation/verify_reference_data.py (CHECK_IPI=1 only) and run only when
PIMD_MACE_MODEL names the model. The engine test long_water_mace_ipi.rs (features torch and
long-validation) runs the same systems with `pimd_cli::run`, in-process MACE (float32, CUDA, bead
chunks), and compares the averages within 5 sigma_combined.

System (identical on both sides): the 64-water box `water_w64` of m8_boxes.json (gen_m8_boxes.py;
192 atoms, cubic L = 12.4295687235 Å; MACE-OFF23 small has r_max = 4.5 Å < L/2), O H H
with masses 15.999, 1.008, 1.008 amu, every bead at the box positions; T = 300 K; P = 1 and P = 8.
Forces: MACE-OFF23 small (the model file of PIMD_MACE_MODEL, as distributed; its SHA-256 and
64-bit FNV-1a are recorded, and the engine test checks the FNV-1a of its own PIMD_MACE_MODEL)
through ONE periodic Python MACE client per i-PI run (../clients/mace_pbc_client.py: ASE
MACECalculator, float32, CUDA, deterministic, profiling executor off, mace's own neighbour list,
started with `pixi run -e ml python`). i-PI's ffsocket has pbc='false' (positions sent unwrapped;
the client wraps them into the cell it receives). The cell is given to i-PI as a diagonal matrix
(mode 'manual'): mode 'abc' gives off-diagonal elements of 1.4e-15 bohr (L cos 90°), which the
client refuses.

Dynamics: i-PI `nvt` with `splitting="baoab"` (B A O A B with the exact free ring-polymer
propagator, the same splitting as SPEC I-1), PILE-L with centroid tau = TAU0_FS and
pile_lambda = 1 (gamma_k = 2 omega_k, SPEC T-2), fixcom = False, dt = 0.5 fs. Both codes use the
same splitting at the same dt, so their discretisation bias is common: bias allowance 0 (brief).
Temperature, time step, tau and masses are given to i-PI in atomic units (k_B T in Hartree, masses
in m_e), with the CODATA 2018 constants of SPEC U-2. i-PI draws its own thermal momenta (SEED);
trajectories differ from the engine's and only averages are compared.

Observables, per molecule (divided by 64), Hartree, from i-PI properties every STRIDE steps:
E_vir = kinetic_cv + potential (SPEC E-2 + E-3), V = potential (bead average, E-3). From the bead
positions every TRAJ_STRIDE steps: g_OO(r), the O-O radial distribution function of SPEC E-7
restricted to the 64 oxygens, averaged over beads: per frame and bead, every O-O pair i < j at the
C-2 minimum-image distance r (orthorhombic, d - L round(d/L)), histogrammed into G_BINS bins of
width G_RMAX/G_BINS on [0, G_RMAX) Å, normalised by n_O (n_O - 1)/2 * (4 pi/3)(r_hi^3 - r_lo^3)/V
(ideal-gas pair count of the shell).

Protocol: N_EQUIL[P] steps discarded, N_PROD[P] production steps. Statistics: block averages with
blocks of BLOCK_STEPS steps (0.5 ps), and of 2 and 4 times that; the stated sigma is the largest of
the three (no assertion on their ratio: with 7 to 10 blocks at 4x the ratio is itself noisy; the
largest of the three is the conservative choice). g_OO per bin likewise.

Run lengths (MILESTONE_8 test 8: "set so that sigma is useful"; spike R6 budget): P = 1: 2.5 ps +
20 ps (45 000 steps); P = 8: 2.5 ps + 15 ps (35 000 steps), at about 50 ms (P = 1) and 0.4 s
(P = 8) per step with one f32 CUDA client, about 3.7 h on the RTX 4060 with the two runs
concurrent on the GPU (the P = 8 run dominates). The P = 1 and P = 8 i-PI runs are run at the same
time, each with its own client.

Reproducibility: i-PI's PRNG is seeded; the client is deterministic on CUDA and stateless (see
its docstring), so the results do not depend on timing; the socket transfers doubles exactly. A
shortened run (200 steps, both P) gave byte-identical JSON twice (checked during test authoring).
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
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

BOX = "water_w64"
MASS_AMU = {"O": 15.999, "H": 1.008}
T_K = 300.0
DT_FS = 0.5
TAU0_FS = 100.0
BEADS = [1, 8]
N_EQUIL = {1: 5000, 8: 5000}
N_PROD = {1: 40000, 8: 30000}
STRIDE = 10
TRAJ_STRIDE = 40
BLOCK_STEPS = 1000
G_RMAX, G_BINS = 6.0, 60  # Angstrom; L/2 = 6.21 Angstrom
SEED = {1: 31415, 8: 92653}
N_WATERS = 64

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
CLIENT = HERE.parent / "clients" / "mace_pbc_client.py"


def conda_version(name):
    meta = Path(sys.prefix) / "conda-meta"
    for f in sorted(meta.glob(f"{name}-*.json")):
        info = json.loads(f.read_text())
        if info.get("name") == name:
            return f"{info['version']} ({info['build']})"
    return "unknown"


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def model_path():
    m = os.environ.get("PIMD_MACE_MODEL")
    if not m or not Path(m).is_file():
        sys.exit("gen_ipi_water_mace.py: set PIMD_MACE_MODEL to the MACE-OFF23 small model file")
    return Path(m).resolve()


def load_box():
    b = json.loads((HERE / "m8_boxes.json").read_text())["boxes"][BOX]
    assert b["n_waters"] == N_WATERS and b["n_atoms"] == 3 * N_WATERS
    assert b["symbols"] == ["O", "H", "H"] * N_WATERS
    return b


def derived(box):
    return {
        "masses_me": [MASS_AMU[s] * AMU for s in box["symbols"]],
        "box_bohr": box["box_angstrom"] / BOHR_A,
        "beta": 1.0 / (T_K * K_B),
        "dt": DT_FS * FS,
        "tau0": TAU0_FS * FS,
    }


def block_sigma(x, length):
    nb = len(x) // length
    assert nb >= 5, nb
    means = x[: nb * length].reshape(nb, length, *x.shape[1:]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(nb)


def stats(x, length):
    s = [block_sigma(x, k * length) for k in (1, 2, 4)]
    return {
        "mean": float(x.mean()),
        "sigma": float(max(s)),
        "sigma_block": float(s[0]),
        "sigma_2block": float(s[1]),
        "sigma_4block": float(s[2]),
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


def g_oo_frames(beads_bohr, box_bohr):
    """Per-frame bead-averaged g_OO: beads_bohr [frames, P, N, 3]; returns [frames, G_BINS]."""
    n_o = N_WATERS
    l_a = box_bohr * BOHR_A
    width = G_RMAX / G_BINS
    lo = np.arange(G_BINS) * width
    shell = 4.0 * np.pi / 3.0 * ((lo + width) ** 3 - lo**3)
    ideal = n_o * (n_o - 1) / 2.0 * shell / l_a**3
    iu, ju = np.triu_indices(n_o, k=1)
    out = np.zeros((beads_bohr.shape[0], G_BINS))
    for f in range(beads_bohr.shape[0]):
        acc = np.zeros(G_BINS)
        for j in range(beads_bohr.shape[1]):
            o = beads_bohr[f, j, 0::3] * BOHR_A  # oxygens, Angstrom
            d = o[ju] - o[iu]
            d -= l_a * np.round(d / l_a)
            r = np.sqrt((d * d).sum(axis=1))
            k = np.floor(r / width).astype(int)
            acc += np.bincount(k[k < G_BINS], minlength=G_BINS)
        out[f] = acc / beads_bohr.shape[1] / ideal
    return out


def write_inputs(tmp, address, p, box, n_steps):
    d = derived(box)
    with open(tmp / "init.xyz", "w") as f:
        f.write(f"{len(box['symbols'])}\n# water_w64, bohr\n")
        for s, x in zip(box["symbols"], box["positions_angstrom"]):
            f.write(f"{s} {x[0] / BOHR_A!r} {x[1] / BOHR_A!r} {x[2] / BOHR_A!r}\n")
    masses = ", ".join(repr(m) for m in d["masses_me"])
    kt = T_K * K_B
    lb = d["box_bohr"]
    (tmp / "input.xml").write_text(
        f"""<simulation verbosity='low' safe_stride='{n_steps}'>
  <output prefix='water'>
    <properties stride='{STRIDE}' filename='out' flush='100'>
      [ step, potential{{atomic_unit}}, kinetic_cv{{atomic_unit}} ]
    </properties>
    <trajectory filename='pos' stride='{TRAJ_STRIDE}' format='xyz' cell_units='atomic_unit' flush='100'>
      positions{{atomic_unit}}
    </trajectory>
  </output>
  <total_steps> {n_steps} </total_steps>
  <prng><seed> {SEED[p]} </seed></prng>
  <ffsocket name='mace' mode='unix' pbc='false'>
    <address> {address} </address>
    <latency> 1e-4 </latency>
    <timeout> 600 </timeout>
  </ffsocket>
  <system>
    <initialize nbeads='{p}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell mode='manual' units='atomic_unit'> [ {lb!r}, 0.0, 0.0, 0.0, {lb!r}, 0.0, 0.0, 0.0, {lb!r} ] </cell>
      <masses mode='manual' units='atomic_unit'> [ {masses} ] </masses>
      <velocities mode='thermal' units='atomic_unit'> {kt!r} </velocities>
    </initialize>
    <forces><force forcefield='mace'/></forces>
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


def run_ipi(tmp, p, box, model, n_steps):
    address = f"pimdmace{os.getpid()}p{p}"
    sock = Path(f"/tmp/ipi_{address}")
    if sock.exists():
        sock.unlink()
    write_inputs(tmp, address, p, box, n_steps)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ipi = subprocess.Popen(
        [shutil.which("i-pi"), "input.xml"], cwd=tmp, stdout=open(tmp / "ipi.log", "w"),
        stderr=subprocess.STDOUT, env=env,
    )
    client = None
    try:
        for _ in range(600):
            if sock.exists():
                break
            assert ipi.poll() is None, (tmp / "ipi.log").read_text()
            time.sleep(0.1)
        client = subprocess.Popen(
            ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", "ml", "python",
             str(CLIENT), "--address", f"unix:{sock}", str(model), "--device", "cuda",
             "--dtype", "float32", "--symbols", *box["symbols"]],
            cwd=tmp, env=env, stdout=open(tmp / "client.log", "w"), stderr=subprocess.STDOUT,
        )
        assert ipi.wait() == 0, (tmp / "ipi.log").read_text()[-3000:]
    finally:
        if client is not None:
            try:
                client.wait(timeout=60)
            except subprocess.TimeoutExpired:
                client.kill()
        if ipi.poll() is None:
            ipi.kill()
    return time.time() - t0


def analyse(tmp, p, box, n_equil, n_prod):
    d = derived(box)
    out = np.loadtxt(tmp / "water.out", comments="#")
    step = out[:, 0].astype(int)
    prod = step > n_equil
    assert prod.sum() == n_prod // STRIDE, prod.sum()
    assert np.all((step[prod] - n_equil) % STRIDE == 0)
    pot, kcv = out[prod, 1], out[prod, 2]
    block = BLOCK_STEPS // STRIDE
    n = len(box["symbols"])
    beads = np.stack(
        [read_xyz_frames(tmp / (f"water.pos_{j}.xyz" if p < 10 else f"water.pos_{j:02d}.xyz"), n)
         for j in range(p)], axis=1,
    )
    frame_steps = np.arange(beads.shape[0]) * TRAJ_STRIDE
    keep = frame_steps > n_equil
    assert keep.sum() == n_prod // TRAJ_STRIDE, keep.sum()
    g = g_oo_frames(beads[keep], d["box_bohr"])
    tblock = BLOCK_STEPS // TRAJ_STRIDE
    s = [block_sigma(g, k * tblock) for k in (1, 2, 4)]
    edges = np.arange(G_BINS + 1) * (G_RMAX / G_BINS)
    return {
        "n_equil": n_equil,
        "n_prod": n_prod,
        "seed": SEED[p],
        "e_vir": stats((kcv + pot) / N_WATERS, block),
        "v": stats(pot / N_WATERS, block),
        "g_oo": {
            "edges_angstrom": edges.tolist(),
            "g": g.mean(axis=0).tolist(),
            "sigma": np.maximum.reduce(s).tolist(),
            "n_frames": int(keep.sum()),
            "block_frames": tblock,
        },
    }


def run_case(p, box, model, n_equil, n_prod, results, times):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        times[p] = run_ipi(tmp, p, box, model, n_equil + n_prod)
        results[p] = analyse(tmp, p, box, n_equil, n_prod)


def generate(n_equil, n_prod):
    model = model_path()
    box = load_box()
    results, times = {}, {}
    threads = [
        threading.Thread(target=run_case, args=(p, box, model, n_equil[p], n_prod[p], results, times))
        for p in BEADS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == BEADS, "an i-PI run failed (see the message above)"
    raw = model.read_bytes()
    return model, raw, box, results, times


def ml_version(module):
    try:
        r = subprocess.run(
            ["pixi", "run", "--manifest-path", str(ROOT / "pixi.toml"), "-e", "ml", "python", "-c",
             f"import {module}; print({module}.__version__)"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    model, raw, box, results, times = generate(N_EQUIL, N_PROD)
    import ipi

    data = {
        "description": "i-PI PIMD of 64 periodic MACE-OFF23 small waters (one Python MACE client, "
        "float32, CUDA): E_vir and V per molecule (Eh), bead-averaged g_OO, block-averaged errors, "
        "P = 1 and P = 8",
        "generator": "gen_ipi_water_mace.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
            "torch (ml)": ml_version("torch"),
            "mace-torch (ml)": ml_version("mace"),
            "run_time_s": {str(p): round(t) for p, t in times.items()},
        },
        "model": {
            "file": model.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fnv1a64": f"{fnv1a64(raw):#018x}",
        },
        "system": {
            "box": BOX,
            "n_waters": N_WATERS,
            "box_angstrom": box["box_angstrom"],
            "masses_amu": MASS_AMU,
            "temperature_kelvin": T_K,
            "dt_fs": DT_FS,
            "tau0_fs": TAU0_FS,
        },
        "derived_atomic_units": {k: v for k, v in derived(box).items() if k != "masses_me"},
        "protocol": {
            "stride": STRIDE,
            "traj_stride": TRAJ_STRIDE,
            "block_steps": BLOCK_STEPS,
            "g_rmax_angstrom": G_RMAX,
            "g_bins": G_BINS,
            "splitting": "baoab",
            "thermostat": "pile_l, pile_lambda = 1, fixcom = False",
            "client": "mace_pbc_client.py, float32, cuda, one client",
        },
        "runs": {f"p{p}": results[p] for p in BEADS},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_water_mace.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; run times {times} s")


if __name__ == "__main__":
    main()
