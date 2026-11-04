"""i-PI reference for the M7 double-well RPMD/TRPMD rung (MILESTONE_7 required test 11; SPEC T-4,
Y-1 ... Y-4, P-2), plus the exact quantum Kubo-transformed C_xx(t) from a DVR (reported, not a gate).

Run from the repository root:

    pixi run -e ref python crates/pimd-estimators/tests/data/gen_ipi_doublewell.py [--out DIR]

Writes ipi_doublewell.json next to this script (or into DIR). The engine test
(long_dynamics_doublewell.rs, feature long-validation) runs the same protocol with the library
(`RingPolymerSim`, `set_thermostat`, `resample_momenta`, `CentroidCorrelation`) and compares
C_xx(tau) and C_vv(tau) at every lag within 5 sigma_combined.

System (identical on both sides, Hartree atomic units, SPEC U-1): one atom of mass m = 1 m_e in the
separable double well of SPEC P-2, V = sum_d (a x_d^4 - b x_d^2), a = 1/8 Eh/bohr^4, b = 1 Eh/bohr^2
(minima at x_d = +-2 bohr, barrier b^2/(4a) = 2 Eh, well frequency sqrt(4b/m) = 2 Eh), beta = 1.5/Eh,
so beta * barrier = 3 (MILESTONE_7 "barrier ~ 3 k_B T") and beta * omega_well = 3. P = 16 beads.
Each Cartesian component is an independent exact 1D problem; C_xx and C_vv sum over x, y, z (Y-4).
Forces: the `doublewell` backend of crates/pimd-socket/tests/clients/ipi_client.py (SPEC P-2,
written from the formula), one client per i-PI instance over a UNIX socket.

Segment protocol (SPEC Y-1 ... Y-4, MILESTONE_7 "For 11 and 12"): i-PI 3.3.1 is driven through
`ipi.scripting.InteractiveSimulation` (one i-PI simulation per chain; each phase of the cycle is
one `run(n)` call on it, the socket and the force cache persist, so switching phases never
re-evaluates the forces, as SPEC I-3/Y-1). The motion is `nvt` with `splitting="baoab"` (the same
B A O A B with the exact free ring-polymer propagator as SPEC I-1/I-2; checked in
ipi/engine/motion/dynamics.py) and a `pile_l` thermostat (ipi/engine/thermostats.py: centroid
Langevin with T = exp(-dt/tau), internal modes tau_k = 1/(2 pile_lambda omega_k), noise at the bead
temperature P T: SPEC T-1/T-2/T-4). `fixcom` is False (SPEC Y-2 does not remove the centre-of-mass
momentum; with one atom i-PI would otherwise zero the centroid momentum). One cycle:
  (a) DECORR steps of PILE-L with tau = TAU0 on the centroid and pile_lambda = 1 (SPEC T-2);
  (b) momentum redraw: every bead momentum from N(0, m P / beta) (SPEC Y-2), drawn from i-PI's
      seeded PRNG (`prng.gvec`), positions and forces unchanged;
  (c) SEG steps of the segment dynamics:
      RPMD:  no O step (the motion's step replaced by the integrator's `mtsprop(0)`, which is
             i-PI's NVE B A A B, identical to BAOAB with the O step removed, SPEC T-3);
      TRPMD: the same PILE-L object with pile_lambda = LAMBDA (tau_k = 1/(2 lambda omega_k), so
             gamma_k = 2 lambda omega_k, SPEC T-4) and tau = +inf on the centroid (T = exp(-0) = 1,
             S = 0: zero friction and zero noise, MILESTONE_7 "a centroid tau of zero friction").
Centroid samples (SPEC Y-3): v = (1/P) sum_j p_j / m and q = (1/P) sum_j q_j at segment steps
s * STRIDE, s = 0 ... SEG/STRIDE (s = 0 right after the redraw), L = SEG/STRIDE + 1 samples.
Per segment C_seg(tau) = (1/(N (L - tau))) sum_s x(s).x(s + tau) (SPEC Y-4). The estimate is the
mean over all segments of all chains; sigma = SD(C_seg, ddof = 1)/sqrt(n_segments) (SPEC Y-4).
Block check (WORKFLOW tolerance policy): sigma also from means of blocks of 4 consecutive segments
within a chain; the generator asserts sigma_4 <= 1.5 sigma_1 at every lag (segments are close to
independent) and stores sigma = max(sigma_1, sigma_4).

Chains: N_CHAINS independent i-PI instances per method, seeds SEED + 100 * method_index + chain,
each N_EQUIL thermostatted steps from all beads at X0, then SEGMENTS_PER_CHAIN cycles. Chains run
in parallel (N_PARALLEL processes); results are combined in a fixed order (method, chain, segment),
so the output does not depend on scheduling. Each chain is deterministic (seeded PRNG, one
single-threaded stateless client, doubles sent exactly over the socket), so verify_reference_data.py
can compare at rtol 1e-9.

DVR (reported in validation/RESULTS.md with the RPMD and TRPMD curves; not a gate): the exact
Kubo-transformed C_xx(t) of the 1D Hamiltonian H = p^2/(2m) + a x^4 - b x^2 at beta,
    Kubo(t) = (1/(Z beta)) sum_{n,m} |x_nm|^2 cos((E_n - E_m) t) w_nm,
    w_nm = (exp(-beta E_m) - exp(-beta E_n))/(E_n - E_m)  (n != m),  beta exp(-beta E_n)  (n = m),
from the eigenpairs of a Colbert-Miller sinc DVR (as gen_double_well.py), times 3 for x, y, z.
Converged in grid size and box (checked here: two grids agree to 1e-8). Also the exact quantum
<|x|^2> (3D) for orientation.

Run time: stated at the end of the run and in the JSON ("wall_s"); about 30 min on the 12-CPU
development machine with N_PARALLEL = 8.
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

MASS = 1.0
A = 0.125
B = 1.0
BETA = 1.5
N_BEADS = 16
DT = 0.05
TAU0 = 1.0
LAMBDA = 0.5
N_EQUIL = 2000
DECORR = 200
SEG = 400
STRIDE = 4
N_LAGS = SEG // STRIDE + 1
X0 = (2.0, -2.0, 2.0)
METHODS = ["rpmd", "trpmd"]
N_CHAINS = 8
SEGMENTS_PER_CHAIN = 200
N_PARALLEL = 8
SEED = 4242


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


def xml_input(address, seed):
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


def worker(method, chain, out):
    """One chain: N_EQUIL thermostatted steps, then SEGMENTS_PER_CHAIN cycles; writes C_seg arrays."""
    from ipi.scripting import InteractiveSimulation
    from ipi.utils.depend import dstrip

    seed = SEED + 100 * METHODS.index(method) + chain
    tmp = Path(tempfile.mkdtemp(prefix=f"dw_{method}_{chain}_"))
    os.chdir(tmp)
    (tmp / "init.xyz").write_text(f"1\n# bohr\nX {X0[0]!r} {X0[1]!r} {X0[2]!r}\n")
    address = f"pimddw{os.getpid()}"
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
    sim = InteractiveSimulation(xml_input(address, seed))
    th.join()
    s = sim.syslist[0]
    motion, thermo = s.motion, s.motion.thermostat
    nve = {"on": False}
    record = {"on": False, "count": 0, "q": [], "v": []}
    m3 = dstrip(s.beads.m3).copy()  # (P, 3N), masses per component

    def sample():
        q = dstrip(s.beads.q).reshape(N_BEADS, -1, 3).mean(axis=0)
        v = (dstrip(s.beads.p) / m3).reshape(N_BEADS, -1, 3).mean(axis=0)
        record["q"].append(q.copy())
        record["v"].append(v.copy())

    def step(self, step=None):
        if nve["on"]:
            self.integrator.mtsprop(0)  # NVE: B A A B, no O step
            self.ensemble.time += self.dt
        else:
            self.integrator.step(step)
            self.ensemble.time += self.dt
        if record["on"]:
            record["count"] += 1
            if record["count"] % STRIDE == 0:
                sample()

    sim.set_motion_step(step)

    def sampling_thermostat():
        nve["on"] = False
        thermo.tau = TAU0
        thermo.pilescale = 1.0

    sampling_thermostat()
    sim.run(N_EQUIL, write_outputs=False)
    cxx, cvv = [], []
    for _ in range(SEGMENTS_PER_CHAIN):
        sampling_thermostat()
        sim.run(DECORR, write_outputs=False)
        # SPEC Y-2: every bead momentum from N(0, m P / beta).
        s.beads.p = sim.prng.gvec((N_BEADS, m3.shape[1])) * np.sqrt(m3 * N_BEADS / BETA)
        if method == "rpmd":
            nve["on"] = True
        else:
            nve["on"] = False
            thermo.tau = np.inf
            thermo.pilescale = LAMBDA
        record.update(on=True, count=0, q=[], v=[])
        sample()
        sim.run(SEG, write_outputs=False)
        record["on"] = False
        assert len(record["q"]) == N_LAGS, len(record["q"])
        cxx.append(c_seg(np.array(record["q"])))
        cvv.append(c_seg(np.array(record["v"])))
    np.savez(out, cxx=np.array(cxx), cvv=np.array(cvv))
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


def dvr(n, half_width):
    """Eigenpairs of the 1D Hamiltonian on a Colbert-Miller sinc DVR of n points on [-L, L]."""
    x = np.linspace(-half_width, half_width, n)
    dx = x[1] - x[0]
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    with np.errstate(divide="ignore"):
        t = np.where(d == 0, np.pi**2 / 3.0, 2.0 / np.where(d == 0, 1, d) ** 2)
    t = t * (-1.0) ** np.abs(d) / (2.0 * MASS * dx * dx)
    h = t + np.diag(A * x**4 - B * x**2)
    e, u = np.linalg.eigh(h)
    return x, e, u


def kubo_cxx(times, n, half_width, n_states=60):
    x, e, u = dvr(n, half_width)
    e, u = e[:n_states], u[:, :n_states]
    xnm = u.T @ (x[:, None] * u)
    e0 = e - e[0]
    w = np.exp(-BETA * e0)
    z = w.sum()
    de = e[:, None] - e[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        wnm = np.where(np.abs(de) > 1e-12, (w[None, :] - w[:, None]) / de, BETA * w[:, None])
    amp = xnm**2 * wnm / (z * BETA)
    out = np.array([(amp * np.cos(de * t)).sum() for t in times])
    x2 = float((w * np.diag(xnm @ xnm)).sum() / z)
    return out, x2, float(e[-1] - e[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--worker", nargs=3, metavar=("METHOD", "CHAIN", "OUT"))
    args = parser.parse_args()
    if args.worker:
        method, chain, out = args.worker
        worker(method, int(chain), out)
        return
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        jobs = [(m, c) for m in METHODS for c in range(N_CHAINS)]
        running, results = [], {}
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        pending = list(jobs)
        while pending or running:
            while pending and len(running) < N_PARALLEL:
                m, c = pending.pop(0)
                out = tmp / f"{m}_{c}.npz"
                log = open(tmp / f"{m}_{c}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", m, str(c), str(out)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=tmp,
                )
                running.append((p, m, c, out))
            time.sleep(1.0)
            for item in list(running):
                p, m, c, out = item
                if p.poll() is not None:
                    assert p.returncode == 0, (tmp / f"{m}_{c}.log").read_text()[-3000:]
                    results[(m, c)] = np.load(out)
                    running.remove(item)
        methods = {}
        for m in METHODS:
            cxx = [results[(m, c)]["cxx"] for c in range(N_CHAINS)]
            cvv = [results[(m, c)]["cvv"] for c in range(N_CHAINS)]
            methods[m] = {
                "n_segments": int(sum(len(c) for c in cxx)),
                "c_xx": stats(cxx),
                "c_vv": stats(cvv),
            }
    wall = time.time() - t0

    lag_t = (np.arange(N_LAGS) * STRIDE * DT).tolist()
    k1, x2_1, top1 = kubo_cxx(lag_t, 301, 6.0)
    k2, x2_2, top2 = kubo_cxx(lag_t, 401, 7.0)
    assert np.max(np.abs(k1 - k2)) < 1e-8, np.max(np.abs(k1 - k2))
    assert BETA * top1 > 40.0, top1  # truncated states carry < exp(-40) weight
    import ipi

    data = {
        "description": "i-PI RPMD and TRPMD (lambda = 1/2) centroid autocorrelations C_xx (bohr^2) "
        "and C_vv ((bohr/(hbar/Eh))^2) of one atom in the SPEC P-2 double well, segment protocol "
        "of SPEC Y-1 ... Y-4; exact quantum Kubo-transformed C_xx(t) from a DVR",
        "generator": "gen_ipi_doublewell.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "i-pi": getattr(ipi, "__version__", conda_version("i-pi")),
            "i-pi_conda": conda_version("i-pi"),
        },
        "system": {
            "mass": MASS,
            "a": A,
            "b": B,
            "beta": BETA,
            "n_beads": N_BEADS,
            "dt": DT,
            "tau0": TAU0,
            "trpmd_lambda": LAMBDA,
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
        },
        "lag_time": lag_t,
        "rpmd": methods["rpmd"],
        "trpmd": methods["trpmd"],
        "exact_kubo_c_xx": (3.0 * k2).tolist(),
        "exact_x2_3d": 3.0 * x2_2,
    }
    data["versions"]["wall_s"] = round(wall)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ipi_doublewell.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path}; i-PI wall time {wall:.0f} s")


if __name__ == "__main__":
    main()
