"""Throughput survey of the M6c study (docs/experiments/m6-water-mace.md, "Protocol" step 5).

    pixi run -e ref python studies/m6-water-mace/bench.py   # from the repository root, machine idle

Time per MD step of the engine (`target/release/pimd-rs`, feature `torch`, inside `pixi run -e
ml`), MACE-OFF23 small, for water and ethanol at P = 1, 8, 16, 32: CPU with 1 and 4 libtorch
threads and CUDA, each in float64 and float32 (CUDA deterministic, the default; plus CUDA float64
with `deterministic = false` at P = 32); and the GFN2-xTB socket path (four `ipi_client.py xtb`
clients) for water at P = 1, 16, 32. Each configuration runs with 100 + 50 steps and with 100 + 1 050
steps (warm-up + production), three times each; ms/step = (best long − best short) / 1 000, so
start-up (model load, CUDA context) and warm-up cancel. Writes results/throughput.csv and results/fig_throughput.png; configs and
outputs under out/bench/ (ignored).
"""

import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "studies/m6-water-mace"
OUT = HERE / "out/bench"
MODEL = "ff_models/exports/m6c-off23-small-f64.pt"
SHORT, LONG, EQUIL, REPEAT = 50, 1050, 100, 3

WATER = (["O", "H", "H"], [15.999, 1.008, 1.008],
         [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
# ase.build.molecule("CH3CH2OH"), as the M6a spike log ("Q3, Q4 correctness").
ETHANOL = (["C", "C", "O", "H", "H", "H", "H", "H", "H"], [12.011, 12.011, 15.999] + [1.008] * 6,
           [[1.168181, -0.400382, 0.0], [0.0, 0.559462, 0.0], [-1.190083, -0.227669, 0.0],
            [-1.946623, 0.381525, 0.0], [0.042557, 1.207508, 0.886933], [0.042557, 1.207508, -0.886933],
            [2.115891, 0.144800, 0.0], [1.128599, -1.037234, 0.885881], [1.128599, -1.037234, -0.885881]])


def toml(system, potential, p, n_steps):
    sym, m, pos = system
    q = lambda v: "[" + ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v) + "]"
    return f"""[system]
n_atoms = {len(sym)}
masses_amu = {q(m)}
symbols = {q(sym)}
positions_angstrom = [{", ".join(q(r) for r in pos)}]

[potential]
{potential}
[sampling]
temperature_kelvin = 300.0
n_beads = {p}
dt_fs = 0.5
tau0_fs = 20.0
n_equil = {EQUIL}
n_steps = {n_steps}
stride = 1
seed = 7

[output]
checkpoint_every = {n_steps}
"""


def mace(device, dtype, threads, deterministic=True):
    return (f'kind = "mace"\nmodel = "{MODEL}"\ndevice = "{device}"\ndtype = "{dtype}"\nthreads = {threads}\n'
            f"deterministic = {'true' if deterministic else 'false'}\n")


def timed(text, threads, xtb):
    """Wall seconds of one engine run of the config `text`."""
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        d = Path(d)
        clients = []
        if xtb:
            sock = d / "s.sock"
            text = text.replace("@ADDRESS@", f"unix:{sock}")
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            clients = [subprocess.Popen([str(ROOT / ".pixi/envs/ref/bin/python"),
                                         str(ROOT / "crates/pimd-socket/tests/clients/ipi_client.py"),
                                         "--address", f"unix:{sock}", "xtb", "--symbols", "O", "H", "H"],
                                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL) for _ in range(4)]
        (d / "c.toml").write_text(text)
        env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
        t0 = time.perf_counter()
        r = subprocess.run(["pixi", "run", "-e", "ml", "env", "LIBTORCH_USE_PYTORCH=1", "target/release/pimd-rs",
                            str(d / "c.toml"), "--out", str(d / "o")], cwd=ROOT, env=env, capture_output=True, text=True)
        t = time.perf_counter() - t0
        for c in clients:
            c.kill()
            c.wait()
        if r.returncode != 0:
            sys.exit(f"engine failed:\n{r.stderr[-2000:]}")
        return t


def cases():
    for name, system in (("water", WATER), ("ethanol", ETHANOL)):
        for p in (1, 8, 16, 32):
            for dtype in ("float64", "float32"):
                for dev, threads in (("cpu", 1), ("cpu", 4), ("cuda", 1)):
                    yield name, system, p, f"mace {dtype} {dev}{threads if dev == 'cpu' else ''}", mace(dev, dtype, threads), threads, False
            if p == 32:
                yield name, system, p, "mace float64 cuda nondet", mace("cuda", "float64", 1, False), 1, False
    for p in (1, 16, 32):
        sock = 'kind = "socket"\naddress = "@ADDRESS@"\nn_clients = 4\ntimeout_s = 120.0\n'
        yield "water", WATER, p, "xtb socket 4 clients", sock, 1, True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load0 = os.getloadavg()
    rows = []
    for name, system, p, backend, pot, threads, xtb in cases():
        short = min(timed(toml(system, pot, p, SHORT), threads, xtb) for _ in range(REPEAT))
        long_ = min(timed(toml(system, pot, p, LONG), threads, xtb) for _ in range(REPEAT))
        ms = (long_ - short) / (LONG - SHORT) * 1000
        rows.append([name, p, backend, f"{ms:.2f}", f"{short:.2f}", f"{long_:.2f}"])
        print(*rows[-1], flush=True)
    load1 = os.getloadavg()
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results/throughput.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# load average at start {load0[0]:.2f}, at end {load1[0]:.2f}"])
        w.writerow(["system", "P", "backend", "ms_per_step", "best_short_s", "best_long_s"])
        w.writerows(rows)
    figure(rows)


def figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ("water", "ethanol")):
        for backend in sorted({r[2] for r in rows if r[0] == name}):
            pts = [(r[1], float(r[3])) for r in rows if r[0] == name and r[2] == backend]
            if len(pts) > 1:
                ax.plot(*zip(*pts), "o-", label=backend)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("P")
        ax.set_title(name)
    axes[0].set_ylabel("ms per MD step")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "results/fig_throughput.png", dpi=150)


if __name__ == "__main__":
    main()
