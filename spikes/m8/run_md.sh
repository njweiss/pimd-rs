#!/usr/bin/env bash
# M8 spike Q3: run one pimd-rs config (kind = "mace") from the repository root.
#   spikes/m8/run_md.sh spikes/m8/configs/<name>.toml [timeout_s]
# Output: spikes/m8/out/md/<name>/ (observables.csv, trajectory.xyz, segments.csv, run.log,
# wall.txt). Engine: target/release/pimd-rs built with --features torch (merged code, unchanged).
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=$1
name=$(basename "$cfg" .toml)
out=spikes/m8/out/md/$name
mkdir -p "$out"
start=$(date +%s.%N)
status=0
OMP_NUM_THREADS=1 timeout "${2:-3600}" target/release/pimd-rs "$cfg" --out "$out" > "$out/run.log" 2>&1 || status=$?
end=$(date +%s.%N)
echo "wall_s=$(echo "$end - $start" | bc) status=$status" | tee "$out/wall.txt"
