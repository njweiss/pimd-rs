"""Produce the engine data for validation/make_plots.py by running the release `pimd-rs` binary.

Run from the repository root (standard library only; any Python 3.9+):

    python3 validation/run_neon.py [--beads 1 4 8 16 32] [--threads 2] [--jobs 5]

For each P it runs validation/configs/neon_p{P}.toml as the chain of long_neon.rs
(crates/pimd-cli/tests/long_neon.rs): 80 legs of 2 500 production steps, the first with
`pimd-rs <cfg> --out leg01`, each later one with `--restart leg{k-1}/checkpoint.bin` and n_steps
raised by 2 500. Restarts are bitwise (tests/restart.rs), so the chain is one uninterrupted
200 000-step trajectory; each leg's observables.csv holds that leg's rows and its rdf.csv the
cumulative g(r), which make_plots.py differences into per-leg block means.

Outputs go to validation/runs/neon_p{P}/leg{NN}/ (gitignored). Older checkpoints are deleted once
the next leg has finished; only the last leg keeps checkpoint.bin. An existing neon_p{P} directory
is replaced. Results do not depend on --threads (D-3).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "validation" / "configs"
RUNS = ROOT / "validation" / "runs"
BIN = ROOT / "target" / "release" / "pimd-rs"
LEGS = 80
LEG_STEPS = 2500


def chain(p, threads):
    base = (CONFIGS / f"neon_p{p}.toml").read_text()
    if not re.search(r"(?m)^n_steps = 200000$", base):
        sys.exit(f"neon_p{p}.toml: expected 'n_steps = 200000'")
    out = RUNS / f"neon_p{p}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    t0 = time.monotonic()
    prev = None
    for leg in range(1, LEGS + 1):
        d = out / f"leg{leg:02d}"
        d.mkdir()
        cfg = d / "config.toml"
        cfg.write_text(re.sub(r"(?m)^n_steps = 200000$", f"n_steps = {leg * LEG_STEPS}", base))
        cmd = [str(BIN), str(cfg), "--out", str(d)]
        if prev is not None:
            cmd += ["--restart", str(prev / "checkpoint.bin")]
        subprocess.run(cmd, env=env, check=True)
        if prev is not None:
            (prev / "checkpoint.bin").unlink()
        prev = d
    dt = time.monotonic() - t0
    print(f"P = {p:2d}: {LEGS} legs in {dt:.0f} s ({threads} threads)", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beads", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--threads", type=int, default=2, help="RAYON_NUM_THREADS per chain")
    ap.add_argument("--jobs", type=int, default=5, help="chains run concurrently")
    a = ap.parse_args()
    subprocess.run(["cargo", "build", "--release", "-p", "pimd-cli"], cwd=ROOT, check=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(lambda p: chain(p, a.threads), a.beads))


if __name__ == "__main__":
    main()
