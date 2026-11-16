"""Write a config and run pimd-rs with K MACE water socket clients (M11 tier 2).

    python studies/m11-dynamics-failure/run_water.py OUT_DIR --t 300 --p 64 --method rpmd \
        [--k 2] [--dt-fs 0.25] [--segments 10] [--seed 2101] [--omega-cm 16000] [--lam 0.5] \
        [--decorr 400] [--seg-steps 8000] [--stride 2] [--n-lags 4001] [--n-equil 800] \
        [--n-mol 64] [--static N_STEPS] [--client-args "--dtype float64 --backend e3nn"]

Run from the repository root with the system python (pimd-rs and the clients are started through
pixi). Writes OUT_DIR/config.toml, runs the engine there, and appends the wall time to
OUT_DIR/wall.txt. --static N runs N plain PIMD steps (no [dynamics]) for timing and static checks.
Methods: classical (P must be 1; RPMD segments), rpmd, trpmd, pa-cmd. Molecules: the MACE
minimum from results/tier2/reference.json, each in a random orientation (fixed seed, independent of
--seed), on a cubic grid of 15 Å; they do not interact (water_client.py). Default step counts give
100 fs decorrelation, 2 ps segments and a 2 ps lag range at Δt = 0.25 fs.
"""

import argparse
import json
import math
import shlex
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
BOHR = 0.529177210903  # Å, U-2
MODEL = os.path.join(ROOT, "ff_models/mace_off23/MACE-OFF23_small.model")
SPACING = 15.0  # Å


def positions(n):
    """n molecules (O, H, H), Å: the reference minimum, randomly rotated (uniform random unit
    quaternions, Python's `random` with a fixed seed), on a cubic grid."""
    import random

    with open(os.path.join(HERE, "results/tier2/reference.json")) as f:
        mol = [[c * BOHR for c in x] for x in json.load(f)["minimum_bohr"]]
    rng = random.Random(2026)
    side = math.ceil(n ** (1 / 3))
    out = []
    for i in range(n):
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        w, x, y, z = (math.sqrt(1 - u1) * math.sin(2 * math.pi * u2), math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                      math.sqrt(u1) * math.sin(2 * math.pi * u3), math.sqrt(u1) * math.cos(2 * math.pi * u3))
        rot = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        c = [SPACING * (i % side), SPACING * ((i // side) % side), SPACING * (i // (side * side))]
        out += [[c[k] + sum(rot[k][m] * atom[m] for m in range(3)) for k in range(3)] for atom in mol]
    return out


def config(a, sock):
    pos = positions(a.n_mol)
    n = 3 * a.n_mol
    lines = [
        f"# M11 tier 2: {a.n_mol} independent H2O (MACE-OFF23 small), {a.method}, T = {a.t} K, P = {a.p} (run_water.py)",
        "[system]",
        f"n_atoms = {n}",
        f"masses_amu = [{', '.join(['15.999', '1.008', '1.008'] * a.n_mol)}]",
        f"symbols = [{', '.join(['\"O\"', '\"H\"', '\"H\"'] * a.n_mol)}]",
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
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--method", choices=["classical", "rpmd", "trpmd", "pa-cmd"], default="rpmd")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--dt-fs", type=float, default=0.25)
    ap.add_argument("--tau0-fs", type=float, default=20.0)
    ap.add_argument("--n-equil", type=int, default=800)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--decorr", type=int, default=400)
    ap.add_argument("--seg-steps", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-lags", type=int, default=4001)
    ap.add_argument("--ckpt-cycles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2101)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--omega-cm", type=float, default=16000.0)
    ap.add_argument("--cmd-lam", type=float, default=1.0)
    ap.add_argument("--n-mol", type=int, default=64)
    ap.add_argument("--client-args", default="", help="extra client arguments")
    ap.add_argument("--python", default="pixi run -e ml python",
                    help="how to start the client's Python (e.g. .venv-cueq/bin/python for cueq)")
    ap.add_argument("--mux", action="store_true",
                    help="one water_mux.py process with P connections (n_clients = P; --k ignored)")
    ap.add_argument("--static", type=int, default=0)
    a = ap.parse_args()
    if a.method == "classical" and a.p != 1:
        ap.error("classical needs --p 1")
    if a.mux:
        a.k = a.p
    os.makedirs(a.out, exist_ok=True)
    sock = tempfile.mktemp(prefix="m11w-", suffix=".sock", dir="/tmp")
    cfg = os.path.join(a.out, "config.toml")
    with open(cfg, "w") as f:
        f.write(config(a, sock))
    if a.mux:
        cmds = [["water_mux.py", "--k", str(a.p)]]
    else:
        cmds = [["water_client.py"]] * a.k
    clients = [subprocess.Popen(
        [*shlex.split(a.python), os.path.join(HERE, cmd[0]), MODEL, *cmd[1:],
         "--config", cfg, "--address", f"unix:{sock}", *shlex.split(a.client_args)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=open(os.path.join(a.out, f"client{c}.log"), "w"),
        stderr=subprocess.STDOUT) for c, cmd in enumerate(cmds)]
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
