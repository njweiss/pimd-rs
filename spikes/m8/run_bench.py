"""M8 spike Q1/Q2: run the bench grid, one process per case, appending JSON lines.

    python3 spikes/m8/run_bench.py GRID OUT.jsonl      (GRID: cuda | cpu | chunk | off24 | ...)

Each case runs `spikes/m8/bench/target/release/m8-bench` under a timeout; a timeout or crash is
recorded as a JSON line with "error". The machine state (load average, nvidia-smi) is recorded
in a "state" line before the grid.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = os.path.join(ROOT, "spikes/m8/bench/target/release/m8-bench")
EXP = "ff_models/exports"
BOX = "spikes/m8/out/boxes"
SKIP_MIB = 6000
MODELS = {
    "small": f"{EXP}/m8-off23-small-f64.pt",
    "medium": f"{EXP}/m8-off23-medium-f64.pt",
    "off24": f"{EXP}/m8-off24-medium-f64.pt",
}


def case(model, nw, p, dtype, device, threads=1, chunk=None, extra=(), timeout=300, budget=30):
    args = [BENCH, "--model", MODELS[model], "--xyz", f"{BOX}/nma_w{nw}.xyz", "--p", str(p),
            "--dtype", dtype, "--device", device, "--threads", str(threads),
            "--budget-s", str(budget)]
    if chunk:
        args += ["--chunk", str(chunk)]
    args += list(extra)
    return dict(label=f"{model} w{nw} P{p} {dtype} {device} t{threads} c{chunk or p} {' '.join(extra)}",
                args=args, threads=threads, timeout=timeout)


def grid(name):
    cs = []
    if name == "cuda":
        for model in ("small", "medium"):
            for dtype in ("f32", "f64"):
                for nw in (64, 128, 216):
                    for p in (1, 8, 16, 32):
                        cs.append(case(model, nw, p, dtype, "cuda"))
    elif name == "cpu":
        for model in ("small", "medium"):
            for dtype in ("f32", "f64"):
                for nw in (64, 128, 216):
                    cs.append(case(model, nw, 1, dtype, "cpu", threads=4, budget=20))
        cs.append(case("small", 64, 8, "f32", "cpu", threads=4, budget=20))
        cs.append(case("small", 64, 8, "f64", "cpu", threads=4, budget=20))
        cs.append(case("small", 216, 1, "f32", "cpu", threads=1, budget=20))
        cs.append(case("small", 216, 1, "f32", "cpu", threads=12, budget=20))
    elif name == "chunk":
        # P = 32, NMA + 216, consecutive bead chunks of c beads (one evaluate per chunk).
        for model, dtype, cs_ in (("small", "f32", (1, 2, 4, 8)), ("medium", "f32", (1, 2)),
                                  ("small", "f64", (1, 2, 4))):
            for c in cs_:
                cs.append(dict(case(model, 216, 32, dtype, "cuda", chunk=c, budget=20), chunked=True))
        cs.append(dict(case("medium", 216, 32, "f64", "cuda", chunk=1, extra=("--warm", "1", "--max-iter", "2"),
                            timeout=500), chunked=True))
        cs.append(dict(case("small", 216, 32, "f32", "cuda", chunk=4, extra=("--nondet",), budget=20), chunked=True))
        cs.append(dict(case("medium", 216, 32, "f32", "cuda", chunk=1, extra=("--nondet",), budget=20), chunked=True))
        # Where the single graph stops fitting (between the fitted and the spilled cases above).
        for model, nw, p in (("small", 216, 10), ("small", 216, 12), ("medium", 216, 2),
                             ("medium", 128, 2), ("medium", 128, 3)):
            cs.append(dict(case(model, nw, p, "f32", "cuda", budget=20, timeout=200), chunked=True))
    elif name == "off24":
        # MACE-OFF24 medium (r_max 6 Å): P = 1 at every N, then P = 32 in chunks of 1 bead.
        for nw in (64, 128, 216):
            cs.append(case("off24", nw, 1, "f32", "cuda"))
        cs.append(dict(case("off24", 216, 32, "f32", "cuda", chunk=1, extra=("--warm", "1", "--max-iter", "3"),
                            budget=20, timeout=600), chunked=True))
    elif name == "expandable":
        # Same cases as the fit limit, with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # (set by main() for this grid): does less allocator slack move the spill point?
        for model, nw, p in (("small", 216, 8), ("small", 216, 10), ("small", 216, 12), ("medium", 216, 2)):
            cs.append(dict(case(model, nw, p, "f32", "cuda", budget=20, timeout=200), chunked=True))
    else:
        raise SystemExit(f"unknown grid {name}")
    return cs


def main():
    name, out = sys.argv[1], sys.argv[2]
    load = open("/proc/loadavg").read().split()[:3]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    with open(out, "a") as f:
        f.write(json.dumps({"state": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "loadavg": load, "nvidia_smi": smi}) + "\n")
    # A case whose smaller-P sibling (same model, dtype, device, N, chunk) failed or needed more
    # than SKIP_MIB of graph memory is skipped: it cannot fit on the 8 GB card.
    big = set()
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            r = json.loads(ln)
            if "label" in r and ("ms_median" in r or "skipped" in r):
                done.add(r["label"])
                if r.get("graph_mem_mib", 0) > SKIP_MIB:
                    big.add(r["label"].split(" P")[0] + r["label"].split(" ", 3)[3].split(" c")[0])
    for c in grid(name):
        key = c["label"].split(" P")[0] + c["label"].split(" ", 3)[3].split(" c")[0]
        if c["label"] in done:
            continue
        if key in big and not c.get("chunked"):
            rec = {"label": c["label"], "skipped": "smaller P did not fit"}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(c["label"], "skipped", flush=True)
            continue
        env = dict(os.environ, OMP_NUM_THREADS=str(c["threads"]))
        if name == "expandable":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        t0 = time.time()
        try:
            r = subprocess.run(c["args"], capture_output=True, text=True, timeout=c["timeout"],
                               cwd=ROOT, env=env)
            line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
            rec = json.loads(line) if line else {"error": f"exit {r.returncode}: {r.stderr[-300:]}"}
            if r.returncode != 0 and "error" not in rec:
                rec["error"] = f"exit {r.returncode}: {r.stderr[-300:]}"
        except subprocess.TimeoutExpired:
            rec = {"error": f"timeout {c['timeout']} s"}
        rec["label"] = c["label"]
        if "error" in rec or rec.get("graph_mem_mib", 0) > SKIP_MIB:
            big.add(key)
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(c["label"], rec.get("ms_median"), rec.get("graph_mem_mib"),
              rec.get("error", "")[:80], f"{rec['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
