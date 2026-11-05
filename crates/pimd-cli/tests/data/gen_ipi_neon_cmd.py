"""i-PI + LAMMPS reference for the M7b liquid-neon PA-CMD rung (MILESTONE_7 M7b required test 10;
SPEC Y-6, Y-7, Y-2 (amended), Y-1 ... Y-5, T-1, T-2, T-4, P-3).

Run from the repository root:

    pixi run -e ref python crates/pimd-cli/tests/data/gen_ipi_neon_cmd.py [--out DIR]

Writes ipi_neon_cmd.json next to this script (or into DIR). The engine test (long_neon_pacmd.rs,
feature long-validation) runs the same protocol with `pimd_cli::run` and `[dynamics] method =
"pa-cmd"` and compares C_vv(tau) at every lag and the Green-Kubo D(tau_max) within 5 sigma_combined.

System: the M4 liquid as gen_ipi_neon_dynamics.py (M7a, locked; same numbers): N = 108 neon atoms,
m = 20.1797 amu, cubic box 14.1 Angstrom, Lennard-Jones eps = 35.6 K, sigma = 2.749 Angstrom,
r_c = 2.2 sigma, shifted, T = 30 K, P = 16, dt = 2 fs, PILE-L tau_0 = 100 fs. Forces: LAMMPS
`fix ipi` with `units electron` (stateless client, see gen_ipi_neon.py). Start: the engine's
pinned lattice, i-PI's own thermal momenta, N_EQUIL PILE-L steps per chain.

PA-CMD (SPEC Y-6, Y-7; MILESTONE_7 M7b decision 8): adiabatic frequency
Omega_h = P^(P/(P-1)) k_B T (hbar = 1) = 16^(16/15) * 30 K * k_B = 401.4 cm^-1 (Omega dt = 0.15),
computed here with the SPEC U-2 constants and stored as system.cmd_omega_cm (the engine test
uses that value as `cmd_omega_cm`); CMD_LAMBDA = 1 (the M7b test value). i-PI 3.3.1
`normal_modes` with `<frequencies style='pa-cmd' units='inversecm'>`. i-PI's cm^-1 factor
4.5563353e-06 Eh is the same angular convention as Omega_au = Omega_cm / HARTREE_IN_WAVENUMBERS,
rounded to 8 digits (i-PI's Omega is ours times 1 + 1.03e-8, stored as ipi_omega_rel_diff). Its
PILE-L uses the dynamical frequency (tau_k = 1/(2 pile_lambda dynomegak)) and the dynamical mass
(thermostats bound to (nm.pnm[k], nm.dynm3[k])), as SPEC Y-6 requires; source references and
run-time assertions as in crates/pimd-estimators/tests/data/gen_ipi_doublewell_cmd.py.

Segment protocol (SPEC Y-7; as gen_ipi_neon_dynamics.py): one `ipi.scripting.InteractiveSimulation`
per chain (motion `nvt`, `splitting="baoab"`, `pile_l`, `fixcom` False, pa-cmd normal modes for the
whole run), each phase one `run(n)` call (no re-evaluation when switching, SPEC I-3). One cycle:
  (a) DECORR steps of PILE-L, tau = TAU0 on the centroid, pile_lambda = 1 (internal gamma = 2 Omega);
  (b) momentum redraw with the dynamical masses (SPEC Y-2 amended): nm.pnm = xi sqrt(dynm3 P/beta)
      from i-PI's seeded PRNG (as i-PI's thermalize_momenta);
  (c) SEG steps of PILE-L with pile_lambda = CMD_LAMBDA and tau = +inf on the centroid (centroid NVE).
Centroid velocities v_i = (1/P) sum_j p_{j,i}/m (SPEC Y-3; the centroid mass is physical) at segment
steps s * STRIDE, s = 0 ... SEG/STRIDE. C_seg(tau) by SPEC Y-4; Green-Kubo per segment
D_seg(tau_n) = (1/3) trapezoid of C_seg on the lag grid (SPEC Y-5). Mean over all segments,
sigma = SD(ddof = 1)/sqrt(n_segments) (SPEC Y-4), with the 4-segment block check of WORKFLOW
(sigma_4 <= 1.5 sigma_1 asserted at every lag; sigma = max).

Chains: N_CHAINS independent i-PI + LAMMPS pairs, seeds SEED + chain, N_PARALLEL at a time,
combined in chain order. Each chain is deterministic (seeded PRNG, one serial stateless LAMMPS
client), so verify_reference_data.py can compare at rtol 1e-9.

Wall time (MILESTONE_7 caps it at 2 h): printed at the end and stored in versions.wall_s (not
compared); 13 min (774 s) on the 12-CPU development machine with N_PARALLEL = 6.
"""

import argparse
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
CMD_LAMBDA = 1.0
# Omega_h = P^(P/(P-1)) k_B T in cm^-1 (SPEC U-2 constants; hbar = 1).
CM_PER_HARTREE = 219474.6313632
OMEGA_CM = N_BEADS ** (N_BEADS / (N_BEADS - 1)) * T_K * K_B * CM_PER_HARTREE
N_EQUIL = 5000
DECORR = 250
SEG = 500
STRIDE = 2
N_LAGS = SEG // STRIDE + 1
N_CHAINS = 6
SEGMENTS_PER_CHAIN = 80
N_PARALLEL = 6
SEED = 16180


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


def c_seg(x):
    """SPEC Y-4 for one segment: x has shape (L, N, 3); returns C_seg(tau), tau = 0 ... N_LAGS - 1."""
    L, n, _ = x.shape
    out = np.empty(N_LAGS)
    for tau in range(N_LAGS):
        out[tau] = np.einsum("sid,sid->", x[: L - tau], x[tau:]) / (n * (L - tau))
    return out


def green_kubo(c, dt_lag):
    """SPEC Y-5: D(tau_n) = (1/3) trapezoid of c on the lag grid, D(0) = 0."""
    d = np.zeros_like(c)
    d[..., 1:] = np.cumsum(0.5 * (c[..., 1:] + c[..., :-1]), axis=-1) * dt_lag / 3.0
    return d


def write_inputs(tmp, address):
    length = BOX_A / BOHR_A
    sigma = SIGMA_A / BOHR_A
    eps = EPS_K * K_B
    rc = CUTOFF_SIGMA * sigma
    skin = SKIN_SIGMA * sigma
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


def xml_input(address, seed):
    length = BOX_A / BOHR_A
    mass = MASS_AMU * AMU
    masses = ", ".join([repr(mass)] * N_ATOMS)
    temp = T_K * K_B
    return f"""<simulation verbosity='quiet' safe_stride='1000000000'>
  <output prefix='neon'>
    <properties stride='1000000000' filename='out'> [ step ] </properties>
  </output>
  <total_steps> 1000000000 </total_steps>
  <prng><seed> {seed} </seed></prng>
  <ffsocket name='lammps' mode='unix' pbc='true'>
    <address> {address} </address>
    <latency> 1e-4 </latency>
  </ffsocket>
  <system>
    <initialize nbeads='{N_BEADS}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell mode='abc' units='atomic_unit'> [ {length!r}, {length!r}, {length!r} ] </cell>
      <masses mode='manual' units='atomic_unit'> [ {masses} ] </masses>
      <velocities mode='thermal' units='atomic_unit'> {temp!r} </velocities>
    </initialize>
    <normal_modes propagator='exact'>
      <frequencies style='pa-cmd' units='inversecm'> [ {OMEGA_CM!r} ] </frequencies>
    </normal_modes>
    <forces><force forcefield='lammps'/></forces>
    <ensemble><temperature units='atomic_unit'> {temp!r} </temperature></ensemble>
    <motion mode='dynamics'>
      <fixcom> False </fixcom>
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


def check_pacmd(s, thermo, pilescale):
    """Asserts i-PI's live PA-CMD objects follow SPEC Y-6 / T-2 / T-4; returns i-PI's Omega (Eh)."""
    from ipi.utils.depend import dstrip

    nm = s.nm
    wk = dstrip(nm.omegak)
    f = dstrip(nm.nm_factor)
    w_ipi = float(nm.nm_freqs[0])
    assert abs(w_ipi / (OMEGA_CM / CM_PER_HARTREE) - 1.0) < 2e-8, w_ipi
    assert f[0] == 1.0, f
    assert np.allclose(f[1:], (wk[1:] / w_ipi) ** 2, rtol=1e-12, atol=0), f
    assert np.allclose(dstrip(nm.dynomegak)[1:], w_ipi, rtol=1e-12, atol=0)
    assert np.allclose(dstrip(nm.dynm3), dstrip(s.beads.m3) * f[:, None], rtol=1e-14, atol=0)
    assert np.allclose(dstrip(thermo.tauk), 1.0 / (2.0 * pilescale * w_ipi), rtol=1e-12, atol=0)
    return w_ipi


def worker(chain, out):
    """One chain: N_EQUIL thermostatted steps, then SEGMENTS_PER_CHAIN cycles; writes C_seg arrays."""
    from ipi.scripting import InteractiveSimulation
    from ipi.utils.depend import dstrip

    seed = SEED + chain
    tmp = Path(tempfile.mkdtemp(prefix=f"neoncmd_{chain}_"))
    os.chdir(tmp)
    address = f"pimdneoncmd{os.getpid()}"
    sock = Path(f"/tmp/ipi_{address}")
    write_inputs(tmp, address)
    beta = 1.0 / (T_K * K_B)
    tau0 = TAU0_FS * FS
    client = []

    def start_client():
        for _ in range(6000):
            if sock.exists():
                break
            time.sleep(0.01)
        client.append(
            subprocess.Popen(
                [shutil.which("lmp"), "-in", "in.lmp", "-log", "none", "-screen", "none",
                 "-nocite"],
                cwd=tmp, stdout=open(tmp / "lmp.log", "w"), stderr=subprocess.STDOUT,
            )
        )

    th = threading.Thread(target=start_client)
    th.start()
    sim = InteractiveSimulation(xml_input(address, seed))
    th.join()
    s = sim.syslist[0]
    thermo = s.motion.thermostat
    record = {"on": False, "count": 0, "v": []}
    m3 = dstrip(s.beads.m3).copy()  # physical masses

    def sample():
        v = (dstrip(s.beads.p) / m3).reshape(N_BEADS, -1, 3).mean(axis=0)
        record["v"].append(v.copy())

    def step(self, step=None):
        self.integrator.step(step)
        self.ensemble.time += self.dt
        if record["on"]:
            record["count"] += 1
            if record["count"] % STRIDE == 0:
                sample()

    sim.set_motion_step(step)

    def sampling_thermostat():
        thermo.tau = tau0
        thermo.pilescale = 1.0

    sampling_thermostat()
    w_ipi = check_pacmd(s, thermo, 1.0)
    sim.run(N_EQUIL, write_outputs=False)
    cvv = []
    for _ in range(SEGMENTS_PER_CHAIN):
        sampling_thermostat()
        sim.run(DECORR, write_outputs=False)
        # SPEC Y-2 (amended): normal-mode momenta from N(0, m s_k P / beta).
        dynm3 = dstrip(s.nm.dynm3)
        s.nm.pnm[:] = sim.prng.gvec((N_BEADS, m3.shape[1])) * np.sqrt(dynm3 * N_BEADS / beta)
        thermo.tau = np.inf
        thermo.pilescale = CMD_LAMBDA
        record.update(on=True, count=0, v=[])
        sample()
        sim.run(SEG, write_outputs=False)
        record["on"] = False
        assert len(record["v"]) == N_LAGS, len(record["v"])
        cvv.append(c_seg(np.array(record["v"])))
    check_pacmd(s, thermo, CMD_LAMBDA)
    np.savez(out, cvv=np.array(cvv), w_ipi=w_ipi)
    for c in client:
        c.kill()
        c.wait()
    os.chdir("/")
    shutil.rmtree(tmp)


def stats(seg_by_chain):
    """Mean, sigma (Y-4) and the 4-segment block check over all chains."""
    allseg = np.concatenate(seg_by_chain)
    n = len(allseg)
    mean = allseg.mean(axis=0)
    s1 = allseg.std(axis=0, ddof=1) / np.sqrt(n)
    blocks = np.concatenate(
        [c[: len(c) // 4 * 4].reshape(-1, 4, c.shape[1]).mean(axis=1) for c in seg_by_chain]
    )
    s4 = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    # D(0) = 0 exactly: no ratio there.
    ok = s1 > 0
    ratio = np.where(ok, s4 / np.where(ok, s1, 1.0), 1.0)
    assert np.all(ratio <= 1.5), float(ratio.max())
    return {
        "mean": mean.tolist(),
        "sigma": np.maximum(s1, s4).tolist(),
        "sigma_segment": s1.tolist(),
        "sigma_4segment": s4.tolist(),
        "max_block_ratio": float(ratio.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--worker", nargs=2, metavar=("CHAIN", "OUT"))
    args = parser.parse_args()
    if args.worker:
        chain, out = args.worker
        worker(int(chain), out)
        return
    t0 = time.time()
    dt_lag = STRIDE * DT_FS * FS
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        pending = list(range(N_CHAINS))
        running, results, w_ipi = [], {}, {}
        while pending or running:
            while pending and len(running) < N_PARALLEL:
                c = pending.pop(0)
                out = tmp / f"pacmd_{c}.npz"
                log = open(tmp / f"pacmd_{c}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(c), str(out)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=tmp,
                )
                running.append((p, c, out))
            time.sleep(1.0)
            for item in list(running):
                p, c, out = item
                if p.poll() is not None:
                    assert p.returncode == 0, (tmp / f"pacmd_{c}.log").read_text()[-3000:]
                    d = np.load(out)
                    results[c] = d["cvv"]
                    w_ipi[c] = float(d["w_ipi"])
                    running.remove(item)
        cvv = [results[c] for c in range(N_CHAINS)]
        dseg = [green_kubo(c, dt_lag) for c in cvv]
        pacmd = {
            "n_segments": int(sum(len(c) for c in cvv)),
            "c_vv": stats(cvv),
            "diffusion": stats(dseg),
        }
    wall = time.time() - t0
    import ipi

    beta = 1.0 / (T_K * K_B)
    data = {
        "description": "i-PI + LAMMPS PA-CMD (normal_modes pa-cmd at Omega_h = P^(P/(P-1)) k_B T, "
        "internal-mode PILE-L pile_lambda = cmd_lambda during segments, centroid NVE) centroid "
        "velocity autocorrelation C_vv ((bohr/(hbar/Eh))^2) and running Green-Kubo D "
        "(bohr^2/(hbar/Eh)) of liquid neon (M4 state point), segment protocol of SPEC Y-7",
        "generator": "gen_ipi_neon_cmd.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
            "lammps_conda": conda_version("lammps"),
            "wall_s": round(wall),
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
            "cmd_omega_cm": OMEGA_CM,
            "cmd_lambda": CMD_LAMBDA,
        },
        "derived_atomic_units": {
            "mass_me": MASS_AMU * AMU,
            "beta": beta,
            "dt_lag": dt_lag,
            "c_vv0_exact": 3.0 / (beta * MASS_AMU * AMU),
            "omega": OMEGA_CM / CM_PER_HARTREE,
            "ipi_omega_rel_diff": w_ipi[0] / (OMEGA_CM / CM_PER_HARTREE) - 1.0,
        },
        "protocol": {
            "n_equil": N_EQUIL,
            "decorrelation_steps": DECORR,
            "segment_steps": SEG,
            "corr_stride": STRIDE,
            "n_lags": N_LAGS,
            "n_chains": N_CHAINS,
            "segments_per_chain": SEGMENTS_PER_CHAIN,
            "seed": SEED,
            "splitting": "baoab",
            "fixcom": False,
            "normal_modes": "pa-cmd, propagator exact, frequency in inversecm",
        },
        "pa_cmd": pacmd,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_neon_cmd.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; i-PI + LAMMPS wall time {wall:.0f} s")


if __name__ == "__main__":
    main()
