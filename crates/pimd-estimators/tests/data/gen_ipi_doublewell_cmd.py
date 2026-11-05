"""i-PI reference for the M7b double-well PA-CMD rung (MILESTONE_7 M7b required test 9; SPEC Y-6,
Y-7, Y-2 (amended), Y-1 ... Y-4, T-1, T-2, T-4, P-2).

Run from the repository root:

    pixi run -e ref python crates/pimd-estimators/tests/data/gen_ipi_doublewell_cmd.py [--out DIR]

Writes ipi_doublewell_cmd.json next to this script (or into DIR). The engine test
(long_pacmd_doublewell.rs, feature long-validation) runs the same protocol through the library
(`RingPolymerIntegrator::with_adiabatic_frequency`, `set_thermostat`, `resample_momenta`,
`CentroidCorrelation`) and compares C_xx(tau) and C_vv(tau) at every lag within 5 sigma_combined,
at both adiabatic frequencies.

System: as gen_ipi_doublewell.py (M7a, locked; same numbers): one atom of mass m = 1 m_e in the
separable SPEC P-2 double well V = sum_d (a x_d^4 - b x_d^2), a = 1/8 Eh/bohr^4, b = 1 Eh/bohr^2,
beta = 1.5/Eh, P = 16 beads, dt = 0.05 hbar/Eh, PILE-L tau_0 = 1 hbar/Eh between segments. Forces:
the `doublewell` backend of crates/pimd-socket/tests/clients/ipi_client.py (SPEC P-2), one client
per i-PI instance over a UNIX socket.

PA-CMD (SPEC Y-6, Y-7; MILESTONE_7 M7b decision 8): adiabatic frequency
Omega_h = P^(P/(P-1)) / beta = 16^(16/15)/1.5 = 12.83 Eh/hbar (Omega dt = 0.64) and 2 Omega_h,
internal-mode friction during segments gamma_k = 2 lambda Omega with lambda = CMD_LAMBDA = 1 (the M7b
test value), centroid NVE. i-PI 3.3.1 `normal_modes` with `<frequencies style='pa-cmd'
units='inversecm'>`, Omega given in cm^-1 as Omega_au * 219474.6313632 (SPEC U-2).

Checked in i-PI 3.3.1's source (2026-11-04):
  * ipi/utils/units.py, UnitMap["frequency"]: "NB Internally, ANGULAR frequencies are used",
    "inversecm": 4.5563353e-06 = 1/219474.63 (Hartree per cm^-1, angular, hbar = 1): the same
    convention as Omega_au = Omega_cm / HARTREE_IN_WAVENUMBERS (no 2 pi). i-PI's factor is rounded
    to 8 digits: i-PI's Omega is ours times (1 + 1.03e-8), recorded in the JSON as
    "ipi_omega_rel_diff" and negligible against the statistical error (~1e-3).
  * ipi/engine/normalmodes.py get_nmm (mode "pa-cmd"): nm_factor[b] = (omegak[b]/Omega)^2 for
    b >= 1, nm_factor[0] = 1 (SPEC Y-6 s_k); get_dynwk: dynomegak = omegak / sqrt(nm_factor)
    (= Omega); get_dynm3: dynm3 = m3 * nm_factor; get_prop_pq: exact free propagator with
    frequency omegak/sqrt(nm_factor) and mass m * nm_factor (SPEC I-2 with Y-6).
  * ipi/engine/thermostats.py ThermoPILE_L: each normal-mode Langevin thermostat is bound to
    (nm.pnm[k], nm.dynm3[k]) (noise in mass-scaled coordinates with the dynamical mass: variance
    (1 - c_k^2) m s_k P / beta, SPEC T-1 with Y-6), and get_tauk = 1 / (2 pilescale dynomegak[k])
    (gamma_k = 2 lambda Omega, SPEC T-2/T-4 with Y-6); the centroid uses tau.
The generator also asserts these at run time on the live objects (nm_factor, dynomegak, tauk).

Segment protocol (SPEC Y-7 = Y-1 with Y-6 masses for the whole run; as gen_ipi_doublewell.py):
`ipi.scripting.InteractiveSimulation`, one i-PI simulation per chain, motion `nvt` with
`splitting="baoab"`, `pile_l`, `fixcom` False, the pa-cmd normal modes from the start (equilibration
included). Each phase is one `run(n)` call (no force re-evaluation when switching, SPEC I-3).
One cycle:
  (a) DECORR steps of PILE-L, tau = TAU0 on the centroid, pile_lambda = 1 (internal gamma = 2 Omega);
  (b) momentum redraw with the dynamical masses (SPEC Y-2 amended): normal-mode momenta
      nm.pnm = xi * sqrt(dynm3 * P / beta), xi from i-PI's seeded PRNG (`prng.gvec`), as i-PI's own
      thermalize_momenta does; positions and forces unchanged. (Statistically identical to the
      engine's draw: an orthogonal transform of iid normals scaled per mode by sqrt(s_k); the
      streams differ anyway.)
  (c) SEG steps of PILE-L with pile_lambda = CMD_LAMBDA and tau = +inf on the centroid
      (T = exp(-0) = 1, S = 0: centroid NVE, SPEC T-4 with Y-6).
Centroid samples (SPEC Y-3, unchanged: s_0 = 1): v = (1/P) sum_j p_j / m and q = (1/P) sum_j q_j at
segment steps s * STRIDE, s = 0 ... SEG/STRIDE. C_seg(tau) by SPEC Y-4; mean over all segments of
all chains; sigma = SD(C_seg, ddof = 1)/sqrt(n_segments); block check with means of 4 consecutive
segments of a chain (sigma_4 <= 1.5 sigma_1 asserted; sigma = max).

Chains: N_CHAINS independent i-PI instances per Omega, seeds SEED + 100 * omega_index + chain, each
N_EQUIL thermostatted steps from all beads at X0, then SEGMENTS_PER_CHAIN cycles; N_PARALLEL at a
time, combined in a fixed order (omega, chain, segment). Each chain is deterministic (seeded PRNG,
one single-threaded stateless client), so verify_reference_data.py can compare at rtol 1e-9.

Run time: printed at the end and stored in versions.wall_s (not compared); 34 min (2 054 s) on the
12-CPU development machine with N_PARALLEL = 8 (MILESTONE_7 caps it at 2 h).
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

ROOT = Path(__file__).resolve().parents[4]
CLIENT = ROOT / "crates/pimd-socket/tests/clients/ipi_client.py"

# SPEC U-2 (CODATA 2018): 1 Eh = 219474.6313632 cm^-1.
CM_PER_HARTREE = 219474.6313632

MASS = 1.0
A = 0.125
B = 1.0
BETA = 1.5
N_BEADS = 16
DT = 0.05
TAU0 = 1.0
CMD_LAMBDA = 1.0
N_EQUIL = 2000
DECORR = 200
SEG = 400
STRIDE = 4
N_LAGS = SEG // STRIDE + 1
X0 = (2.0, -2.0, 2.0)
OMEGA_H = N_BEADS ** (N_BEADS / (N_BEADS - 1)) / BETA
OMEGAS = {"omega_h": OMEGA_H, "omega_2h": 2.0 * OMEGA_H}
KEYS = list(OMEGAS)
N_CHAINS = 8
SEGMENTS_PER_CHAIN = 200
N_PARALLEL = 8
SEED = 5151


def conda_version(name):
    meta = Path(sys.prefix) / "conda-meta"
    for f in sorted(meta.glob(f"{name}-*.json")):
        info = json.loads(f.read_text())
        if info.get("name") == name:
            return f"{info['version']} ({info['build']})"
    return "unknown"


def c_seg(x):
    """SPEC Y-4 for one segment: x has shape (L, N, 3); returns C_seg(tau), tau = 0 ... N_LAGS - 1."""
    L, n, _ = x.shape
    out = np.empty(N_LAGS)
    for tau in range(N_LAGS):
        out[tau] = np.einsum("sid,sid->", x[: L - tau], x[tau:]) / (n * (L - tau))
    return out


def xml_input(address, seed, omega):
    temp = 1.0 / BETA
    return f"""<simulation verbosity='quiet' safe_stride='1000000000'>
  <output prefix='dw'>
    <properties stride='1000000000' filename='out'> [ step ] </properties>
  </output>
  <total_steps> 1000000000 </total_steps>
  <prng><seed> {seed} </seed></prng>
  <ffsocket name='dw' mode='unix' pbc='false'>
    <address> {address} </address>
    <latency> 1e-6 </latency>
  </ffsocket>
  <system>
    <initialize nbeads='{N_BEADS}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell mode='abc' units='atomic_unit'> [ 100.0, 100.0, 100.0 ] </cell>
      <masses mode='manual' units='atomic_unit'> [ {MASS!r} ] </masses>
      <velocities mode='thermal' units='atomic_unit'> {temp!r} </velocities>
    </initialize>
    <normal_modes propagator='exact'>
      <frequencies style='pa-cmd' units='inversecm'> [ {omega * CM_PER_HARTREE!r} ] </frequencies>
    </normal_modes>
    <forces><force forcefield='dw'/></forces>
    <ensemble><temperature units='atomic_unit'> {temp!r} </temperature></ensemble>
    <motion mode='dynamics'>
      <fixcom> False </fixcom>
      <dynamics mode='nvt' splitting='baoab'>
        <timestep units='atomic_unit'> {DT!r} </timestep>
        <thermostat mode='pile_l'>
          <tau units='atomic_unit'> {TAU0!r} </tau>
          <pile_lambda> 1.0 </pile_lambda>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
"""


def check_pacmd(s, thermo, omega, pilescale):
    """Asserts i-PI's live PA-CMD objects follow SPEC Y-6 / T-2 / T-4; returns i-PI's Omega."""
    from ipi.utils.depend import dstrip

    nm = s.nm
    wk = dstrip(nm.omegak)
    f = dstrip(nm.nm_factor)
    w_ipi = float(nm.nm_freqs[0])
    assert abs(w_ipi / omega - 1.0) < 2e-8, (w_ipi, omega)
    assert f[0] == 1.0, f
    assert np.allclose(f[1:], (wk[1:] / w_ipi) ** 2, rtol=1e-12, atol=0), f
    assert np.allclose(dstrip(nm.dynomegak)[1:], w_ipi, rtol=1e-12, atol=0)
    assert np.allclose(dstrip(nm.dynm3), dstrip(s.beads.m3) * f[:, None], rtol=1e-14, atol=0)
    assert np.allclose(dstrip(thermo.tauk), 1.0 / (2.0 * pilescale * w_ipi), rtol=1e-12, atol=0)
    return w_ipi


def worker(key, chain, out):
    """One chain: N_EQUIL thermostatted steps, then SEGMENTS_PER_CHAIN cycles; writes C_seg arrays."""
    from ipi.scripting import InteractiveSimulation
    from ipi.utils.depend import dstrip

    omega = OMEGAS[key]
    seed = SEED + 100 * KEYS.index(key) + chain
    tmp = Path(tempfile.mkdtemp(prefix=f"dwcmd_{key}_{chain}_"))
    os.chdir(tmp)
    (tmp / "init.xyz").write_text(f"1\n# bohr\nX {X0[0]!r} {X0[1]!r} {X0[2]!r}\n")
    address = f"pimddwcmd{os.getpid()}"
    sock = Path(f"/tmp/ipi_{address}")
    client = []

    def start_client():
        for _ in range(6000):
            if sock.exists():
                break
            time.sleep(0.01)
        client.append(
            subprocess.Popen(
                [sys.executable, str(CLIENT), "--address", f"unix:{sock}", "--pbc", "none",
                 "doublewell", "--a", repr(A), "--b", repr(B)]
            )
        )

    th = threading.Thread(target=start_client)
    th.start()
    sim = InteractiveSimulation(xml_input(address, seed, omega))
    th.join()
    s = sim.syslist[0]
    thermo = s.motion.thermostat
    record = {"on": False, "count": 0, "q": [], "v": []}
    m3 = dstrip(s.beads.m3).copy()  # (P, 3N), physical masses per component

    def sample():
        q = dstrip(s.beads.q).reshape(N_BEADS, -1, 3).mean(axis=0)
        v = (dstrip(s.beads.p) / m3).reshape(N_BEADS, -1, 3).mean(axis=0)
        record["q"].append(q.copy())
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
        thermo.tau = TAU0
        thermo.pilescale = 1.0

    sampling_thermostat()
    w_ipi = check_pacmd(s, thermo, omega, 1.0)
    sim.run(N_EQUIL, write_outputs=False)
    cxx, cvv = [], []
    for _ in range(SEGMENTS_PER_CHAIN):
        sampling_thermostat()
        sim.run(DECORR, write_outputs=False)
        # SPEC Y-2 (amended): normal-mode momenta from N(0, m s_k P / beta).
        dynm3 = dstrip(s.nm.dynm3)
        s.nm.pnm[:] = sim.prng.gvec((N_BEADS, m3.shape[1])) * np.sqrt(dynm3 * N_BEADS / BETA)
        thermo.tau = np.inf
        thermo.pilescale = CMD_LAMBDA
        record.update(on=True, count=0, q=[], v=[])
        sample()
        sim.run(SEG, write_outputs=False)
        record["on"] = False
        assert len(record["q"]) == N_LAGS, len(record["q"])
        cxx.append(c_seg(np.array(record["q"])))
        cvv.append(c_seg(np.array(record["v"])))
    check_pacmd(s, thermo, omega, CMD_LAMBDA)
    np.savez(out, cxx=np.array(cxx), cvv=np.array(cvv), w_ipi=w_ipi)
    for c in client:
        c.kill()
        c.wait()
    os.chdir("/")
    shutil.rmtree(tmp)


def stats(cseg_by_chain):
    """Mean, sigma (Y-4) and the 4-segment block check over all chains."""
    allseg = np.concatenate(cseg_by_chain)
    n = len(allseg)
    mean = allseg.mean(axis=0)
    s1 = allseg.std(axis=0, ddof=1) / np.sqrt(n)
    blocks = np.concatenate(
        [c[: len(c) // 4 * 4].reshape(-1, 4, c.shape[1]).mean(axis=1) for c in cseg_by_chain]
    )
    s4 = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    ratio = s4 / s1
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
    parser.add_argument("--worker", nargs=3, metavar=("OMEGA", "CHAIN", "OUT"))
    args = parser.parse_args()
    if args.worker:
        key, chain, out = args.worker
        worker(key, int(chain), out)
        return
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        jobs = [(k, c) for k in KEYS for c in range(N_CHAINS)]
        running, results = [], {}
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        pending = list(jobs)
        while pending or running:
            while pending and len(running) < N_PARALLEL:
                k, c = pending.pop(0)
                out = tmp / f"{k}_{c}.npz"
                log = open(tmp / f"{k}_{c}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", k, str(c), str(out)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=tmp,
                )
                running.append((p, k, c, out))
            time.sleep(1.0)
            for item in list(running):
                p, k, c, out = item
                if p.poll() is not None:
                    assert p.returncode == 0, (tmp / f"{k}_{c}.log").read_text()[-3000:]
                    results[(k, c)] = np.load(out)
                    running.remove(item)
        runs = {}
        for k in KEYS:
            cxx = [results[(k, c)]["cxx"] for c in range(N_CHAINS)]
            cvv = [results[(k, c)]["cvv"] for c in range(N_CHAINS)]
            w_ipi = float(results[(k, 0)]["w_ipi"])
            runs[k] = {
                "omega": OMEGAS[k],
                "omega_cm": OMEGAS[k] * CM_PER_HARTREE,
                "ipi_omega_rel_diff": w_ipi / OMEGAS[k] - 1.0,
                "n_segments": int(sum(len(c) for c in cxx)),
                "c_xx": stats(cxx),
                "c_vv": stats(cvv),
            }
    wall = time.time() - t0
    import ipi

    lag_t = (np.arange(N_LAGS) * STRIDE * DT).tolist()
    data = {
        "description": "i-PI PA-CMD (normal_modes pa-cmd, internal-mode PILE-L pile_lambda = "
        "cmd_lambda during segments, centroid NVE) centroid autocorrelations C_xx (bohr^2) and "
        "C_vv ((bohr/(hbar/Eh))^2) of one atom in the SPEC P-2 double well at Omega_h = "
        "P^(P/(P-1))/beta and 2 Omega_h, segment protocol of SPEC Y-7",
        "generator": "gen_ipi_doublewell_cmd.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
            "wall_s": round(wall),
        },
        "system": {
            "mass": MASS,
            "a": A,
            "b": B,
            "beta": BETA,
            "n_beads": N_BEADS,
            "dt": DT,
            "tau0": TAU0,
            "cmd_lambda": CMD_LAMBDA,
            "x0": list(X0),
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
        "lag_time": lag_t,
        "omega_h": runs["omega_h"],
        "omega_2h": runs["omega_2h"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_doublewell_cmd.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; i-PI wall time {wall:.0f} s")


if __name__ == "__main__":
    main()
