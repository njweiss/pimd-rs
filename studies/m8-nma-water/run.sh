#!/usr/bin/env bash
# Run one M8b config (docs/experiments/m8-nma-water.md) from the repository root:
#   studies/m8-nma-water/run.sh studies/m8-nma-water/configs/<set>/<run>.toml [--restart]
# Output: studies/m8-nma-water/out/<set>/<run>/ (observables.csv, trajectory.xyz, checkpoint.bin,
# run.log, config.toml as run, gpu.csv sampled every 5 s, wall.txt appended per invocation).
# --restart continues from that directory's checkpoint.bin (bitwise, D-3). A restart overwrites
# observables.csv, trajectory.xyz and run.log, so the earlier outputs are first moved to part<k>/
# (k = 0, 1, ...); they hold rows past the checkpoint step, which the analysis drops when it joins
# the parts. The engine is the release build of pimd-cli with feature `torch`, run inside the pixi
# `ml` env.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=$1
name=$(basename "$cfg" .toml)
set_=$(basename "$(dirname "$cfg")")
out=studies/m8-nma-water/out/$set_/$name
mkdir -p "$out"
restart=()
if [ "${2:-}" = "--restart" ]; then
  restart=(--restart "$out/checkpoint.bin")
  cmp -s "$cfg" "$out/config.toml" || { echo "config differs from $out/config.toml" >&2; exit 1; }
  k=0
  while [ -e "$out/part$k" ]; do k=$((k + 1)); done
  mkdir "$out/part$k"
  for f in observables.csv trajectory.xyz run.log stdout.log; do
    [ ! -e "$out/$f" ] || mv "$out/$f" "$out/part$k/"
  done
else
  [ ! -e "$out/checkpoint.bin" ] || { echo "$out has a checkpoint; use --restart" >&2; exit 1; }
  cp "$cfg" "$out/config.toml"
fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 >> "$out/gpu.csv" &
smi=$!
trap 'kill $smi 2>/dev/null || true' EXIT
start=$(date +%s.%N)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAYON_NUM_THREADS=4 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 \
  target/release/pimd-rs "$out/config.toml" --out "$out" "${restart[@]}" >> "$out/stdout.log" 2>&1
end=$(date +%s.%N)
printf 'wall_s %s restart %s commit %s\n' "$(echo "$end - $start" | bc)" "${#restart[@]}" \
  "$(git rev-parse --short HEAD)" | tee -a "$out/wall.txt"
