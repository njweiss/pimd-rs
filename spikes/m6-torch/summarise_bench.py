"""Markdown tables from run_bench.sh output (spike question 5).

    python3 spikes/m6-torch/summarise_bench.py spikes/m6-torch/out/q5.log

Per model and system: median ms per evaluation of all P beads (batched), and in brackets the
per-bead-loop time where measured; one column per (dtype, device). Standard library only.
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(\S+) (\S+) P=(\d+) (f64|f32) (cpu1|cpu4|cuda) (batched|loop) \| median ([\d.]+) ms")
COLS = [(d, v) for d in ("f64", "f32") for v in ("cpu1", "cpu4", "cuda")]


def main():
    t = defaultdict(dict)
    for ln in open(sys.argv[1]):
        m = LINE.match(ln)
        if m:
            model, sys_, p, dt, dev, mode, ms = m.groups()
            t[(model, sys_)][(int(p), dt, dev, mode)] = float(ms)
    for (model, sys_), d in t.items():
        print(f"\n**{model}, {sys_}**: ms per evaluation of all beads, batched (per-bead loop)\n")
        print("| P | " + " | ".join(f"{dt} {dev}" for dt, dev in COLS) + " |")
        print("|---" * (len(COLS) + 1) + "|")
        for p in sorted({k[0] for k in d}):
            cells = []
            for dt, dev in COLS:
                b = d.get((p, dt, dev, "batched"))
                lo = d.get((p, dt, dev, "loop"))
                c = "" if b is None else f"{b:.1f}"
                if lo is not None:
                    c += f" ({lo:.1f})"
                cells.append(c)
            print(f"| {p} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
