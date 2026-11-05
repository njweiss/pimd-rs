"""M8 spike: markdown tables from the bench JSON lines.

    python3 spikes/m8/summarise.py spikes/m8/out/bench_*.jsonl
"""

import json
import sys

rows = []
for path in sys.argv[1:]:
    for ln in open(path):
        r = json.loads(ln)
        if "label" in r:
            rows.append(r)

print("| model | dtype | dev | thr | atoms | P | chunk | edges/bead | ms/eval | ms/bead | µs/edge | "
      "graph MiB | kB/edge | edges all-pairs ms | cell list ms | edge share | note |")
print("|" + "---|" * 17)
for r in rows:
    lab = r["label"].split()
    model = lab[0]
    if "skipped" in r:
        print(f"| {model} | {lab[3]} | {lab[4]} | {lab[5][1:]} | {lab[1]} | {lab[2][1:]} | {lab[6][1:]} "
              f"| | | | | | | | | | skipped (smaller P > 6 GB) |")
        continue
    if "ms_median" not in r:
        print(f"| {model} | {lab[3]} | {lab[4]} | {lab[5][1:]} | {lab[1]} | {lab[2][1:]} | {lab[6][1:]} "
              f"| {r.get('edges_per_bead', '')} | | | | | | | | | {r.get('error', '')[:60]} |")
        continue
    gm = r["graph_mem_mib"] if r["device"] == "cuda" else None
    kb = gm * 1024 / (r["edges_per_bead"] * min(r["chunk"], r["p"])) if gm is not None else None
    note = " ".join(lab[7:])
    if not r.get("deterministic", True):
        note = "nondet"
    if not r["celllist_equal"]:
        note += " CELL LIST DIFFERS"
    print(f"| {model} | {r['dtype']} | {r['device']} | {r['threads']} | {r['n_atoms']} | {r['p']} | "
          f"{r['chunk']} | {r['edges_per_bead']} | {r['ms_median']:.1f} | {r['ms_per_bead']:.1f} | "
          f"{r['us_per_edge']:.2f} | {'' if gm is None else gm} | {'' if kb is None else f'{kb:.1f}'} | "
          f"{r['t_edges_allpairs_ms']:.2f} | {r['t_edges_celllist_ms']:.2f} | "
          f"{100 * r['edge_share']:.2f} % | {note} |")
