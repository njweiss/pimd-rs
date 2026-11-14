#!/usr/bin/env bash
# M10 spike Q3 (v): the M8b pilot runs (short and long) through the merged socket backend with
# K cueq_client.py clients on the one GPU, for ms per step = Δwall / Δsteps.
#   spikes/m10/q3v.sh P K [--e3nn]
# Output: spikes/m10/out/q3v/P<P>_k<K>[_e3nn]/{<run>/, wall.txt, client*.log}.
set -euo pipefail
cd "$(dirname "$0")/../.."
p=$1; k=$2; flag=${3:-}
tag=P${p}_k${k}${flag:+_e3nn}
out=spikes/m10/out/q3v/$tag
rm -rf "$out"; mkdir -p "$out"
{ date -Is; uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; git rev-parse --short HEAD; } >> "$out/wall.txt"
for n in 40 200; do
  run=p${p}_n$n
  d=$out/$run; mkdir -p "$d"
  sock=$(mktemp -u /tmp/m10-XXXXXX.sock)
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
    pixi run -e ml spikes/m10/cueq/.venv/bin/python spikes/m10/cueq_client.py --address "unix:$sock" $flag \
      < /dev/null > "$out/client_${n}_$c.log" 2>&1 &
    pids+=($!)
  done
  s=$(date +%s.%N)
  RAYON_NUM_THREADS=4 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 target/release/pimd-rs "$d/config.toml" --out "$d" > "$d/stdout.log" 2>&1 || echo "$run engine failed" | tee -a "$out/wall.txt"
  e=$(date +%s.%N)
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  echo "$run wall_s $(echo "$e - $s" | bc)" | tee -a "$out/wall.txt"
done
