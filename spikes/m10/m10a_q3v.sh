#!/usr/bin/env bash
# M10a measurement (MILESTONE_10 "Acceptance"): the spike's Q3 (v) with the supported client
# crates/pimd-torch/scripts/mace_client.py (cueq, float32, CUDA) from .venv-cueq/ (README), and the
# same pilot config in-process for the energy comparison. Adapted from q3v.sh.
#   spikes/m10/m10a_q3v.sh P K
# Needs target/release/pimd-rs built with feature `torch`.
# Output: spikes/m10/out/m10a/P<P>_k<K>/{<run>/, wall.txt, client*.log}, spikes/m10/out/m10a/inproc/.
set -euo pipefail
cd "$(dirname "$0")/../.."
p=$1; k=$2
out=spikes/m10/out/m10a/P${p}_k${k}
rm -rf "$out"; mkdir -p "$out"
{ date -Is; uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; git rev-parse --short HEAD; } >> "$out/wall.txt"
for n in 40 200; do
  run=p${p}_n$n
  d=$out/$run; mkdir -p "$d"
  sock=$(mktemp -u /tmp/m10a-XXXXXX.sock)
  python3 - "$run" "$sock" "$k" "$d/config.toml" <<'EOF'
import re, sys
run, sock, k, dst = sys.argv[1:]
s = open(f"studies/m8-nma-water/configs/pilot/{run}.toml").read()
pot = f'[potential]\nkind = "socket"\naddress = "unix:{sock}"\nn_clients = {k}\ntimeout_s = 600.0\n\n'
s = re.sub(r"\[potential\]\n.*?\n\n", pot, s, flags=re.S)
open(dst, "w").write(s)
EOF
  pids=()
  for c in $(seq 1 "$k"); do
    pixi run -e ml .venv-cueq/bin/python crates/pimd-torch/scripts/mace_client.py \
      ff_models/mace_off23/MACE-OFF23_small.model --config "$d/config.toml" --address "unix:$sock" \
      --backend cueq --dtype float32 --device cuda < /dev/null > "$out/client_${n}_$c.log" 2>&1 &
    pids+=($!)
  done
  s=$(date +%s.%N)
  RAYON_NUM_THREADS=4 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 target/release/pimd-rs "$d/config.toml" --out "$d" > "$d/stdout.log" 2>&1 || echo "$run engine failed" | tee -a "$out/wall.txt"
  e=$(date +%s.%N)
  for pid in "${pids[@]}"; do wait "$pid" || echo "client $pid exit $?" | tee -a "$out/wall.txt"; done
  echo "$run wall_s $(echo "$e - $s" | bc)" | tee -a "$out/wall.txt"
done
# In-process reference: the unchanged pilot config (export m8b-off23-small-f64.pt, float32, CUDA, c = 8).
d=spikes/m10/out/m10a/inproc/p${p}_n40
rm -rf "$d"; mkdir -p "$d"
pixi run -e ml env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAYON_NUM_THREADS=4 LIBTORCH_USE_PYTORCH=1 \
  target/release/pimd-rs studies/m8-nma-water/configs/pilot/p${p}_n40.toml --out "$d" > "$d/stdout.log" 2>&1
