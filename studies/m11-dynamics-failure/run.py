"""Write a config and run pimd-rs with K Morse socket clients (M11 tier 1).

    python studies/m11-dynamics-failure/run.py OUT_DIR --system s1 --t 300 --p 32 --method rpmd \
        [--k 2] [--dt-fs 0.1] [--segments 100] [--seed 1101] [--omega-cm 16000] [--lam 0.5] \
        [--decorr 1000] [--seg-steps 20000] [--stride 2] [--n-lags 10001] [--n-equil 2000] \
        [--n-atoms 64] [--static N_STEPS]

Run from the repository root with the system python (pimd-rs and the clients are started through
pixi). Writes OUT_DIR/config.toml, runs the engine there, and appends the wall time to
OUT_DIR/wall.txt. --static N runs N plain PIMD steps (no [dynamics]) for timing and static checks.
Methods: classical (P must be 1; RPMD segments), rpmd, trpmd, pa-cmd.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
MU_AMU = 15.999 * 1.008 / (15.999 + 1.008)
R_E_ANGSTROM = 1.8324 * 0.529177210903


def positions(system, n):
    """S1: every component at r_e. S2: r_e along n fixed, spread directions (golden spiral)."""
    if system == "s1":
        return [[R_E_ANGSTROM] * 3 for _ in range(n)]
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        rho = math.sqrt(1.0 - z * z)
        phi = i * math.pi * (3.0 - math.sqrt(5.0))
        out.append([R_E_ANGSTROM * rho * math.cos(phi), R_E_ANGSTROM * rho * math.sin(phi),
                    R_E_ANGSTROM * z])
    return out


def config(a, sock):
    pos = positions(a.system, a.n_atoms)
    lines = [
        f"# M11 tier 1: {a.system} Morse OH, {a.method}, T = {a.t} K, P = {a.p} (run.py)",
        "[system]",
        f"n_atoms = {a.n_atoms}",
        f"masses_amu = [{', '.join([repr(MU_AMU)] * a.n_atoms)}]",
        f"positions_angstrom = [{', '.join('[' + ', '.join(repr(c) for c in r) + ']' for r in pos)}]",
        "",
        "[potential]",
        'kind = "socket"',
        f'address = "unix:{sock}"',
        f"n_clients = {a.k}",
        "timeout_s = 600.0",
        "",
        "[sampling]",
        f"temperature_kelvin = {float(a.t)}",
        f"n_beads = {a.p}",
        f"dt_fs = {a.dt_fs}",
        f"tau0_fs = {a.tau0_fs}",
        f"n_equil = {a.n_equil}",
    ]
    if a.static:
        lines += [f"n_steps = {a.static}", "stride = 1", f"seed = {a.seed}", "",
                  "[output]", f"checkpoint_every = {a.static}", f"trajectory_every = {a.static}"]
        return "\n".join(lines) + "\n"
    cycle = a.decorr + a.seg_steps
    n_steps = cycle * a.segments
    lines += [f"n_steps = {n_steps}", "stride = 10", f"seed = {a.seed}", "", "[dynamics]"]
    if a.method in ("classical", "rpmd"):
        lines.append('method = "rpmd"')
    elif a.method == "trpmd":
        lines += ['method = "trpmd"', f"trpmd_lambda = {a.lam}"]
    else:
        lines += ['method = "pa-cmd"', f"cmd_omega_cm = {a.omega_cm}", f"cmd_lambda = {a.cmd_lam}"]
    lines += [f"decorrelation_steps = {a.decorr}", f"segment_steps = {a.seg_steps}",
              f"corr_stride = {a.stride}", f"n_lags = {a.n_lags}", "",
              "[output]", f"checkpoint_every = {cycle * a.ckpt_cycles}", f"trajectory_every = {n_steps}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--system", choices=["s1", "s2"], required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--method", choices=["classical", "rpmd", "trpmd", "pa-cmd"], default="rpmd")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--dt-fs", type=float, default=0.1)
    ap.add_argument("--tau0-fs", type=float, default=20.0)
    ap.add_argument("--n-equil", type=int, default=2000)
    ap.add_argument("--segments", type=int, default=100)
    ap.add_argument("--decorr", type=int, default=1000)
    ap.add_argument("--seg-steps", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-lags", type=int, default=10001)
    ap.add_argument("--ckpt-cycles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1101)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--omega-cm", type=float, default=16000.0)
    ap.add_argument("--cmd-lam", type=float, default=1.0)
    ap.add_argument("--n-atoms", type=int, default=64)
    ap.add_argument("--static", type=int, default=0)
    a = ap.parse_args()
    if a.method == "classical" and a.p != 1:
        ap.error("classical needs --p 1")
    os.makedirs(a.out, exist_ok=True)
    sock = tempfile.mktemp(prefix="m11-", suffix=".sock", dir="/tmp")
    cfg = os.path.join(a.out, "config.toml")
    with open(cfg, "w") as f:
        f.write(config(a, sock))
    clients = [subprocess.Popen(
        ["pixi", "run", "-e", "ref", "python", os.path.join(HERE, "morse_client.py"),
         "--address", f"unix:{sock}", a.system],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=open(os.path.join(a.out, f"client{c}.log"), "w"),
        stderr=subprocess.STDOUT) for c in range(a.k)]
    t0 = time.time()
    with open(os.path.join(a.out, "stdout.log"), "w") as log:
        rc = subprocess.call([os.path.join(ROOT, "target/release/pimd-rs"), cfg, "--out", a.out],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, RAYON_NUM_THREADS="1"))
    wall = time.time() - t0
    for c in clients:
        try:
            c.wait(timeout=30)
        except subprocess.TimeoutExpired:
            c.kill()
    with open(os.path.join(a.out, "wall.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rc {rc} wall_s {wall:.3f}\n")
    print(f"{a.out}: rc {rc}, wall {wall:.1f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
