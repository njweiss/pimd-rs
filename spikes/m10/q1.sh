#!/usr/bin/env bash
# M10 spike Q1 (docs/experiments/m10-profiling-spike.md): engine share of a production step.
# Runs the M8b pilot configs (short and long at P = 8 and 32, c = 8) with the merged pimd-rs, then
# the M8 bench (MacePotential::evaluate alone, same box, same c) in the same environment as M8b
# production (run.sh). Output: spikes/m10/out/q1/{wall.txt,bench.jsonl,<run>/}.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=spikes/m10/out/q1
mkdir -p "$out"
env_=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAYON_NUM_THREADS=4 LIBTORCH_USE_PYTORCH=1)
{ date -Is; uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; git rev-parse --short HEAD; } >> "$out/wall.txt"
for run in p8_n40 p8_n200 p32_n40 p32_n200; do
  d=$out/$run
  rm -rf "$d"; mkdir -p "$d"
  s=$(date +%s.%N)
  pixi run -e ml "${env_[@]}" target/release/pimd-rs studies/m8-nma-water/configs/pilot/$run.toml --out "$d" > "$d/stdout.log" 2>&1
  e=$(date +%s.%N)
  echo "$run wall_s $(echo "$e - $s" | bc)" | tee -a "$out/wall.txt"
done
for p in 8 32; do
  pixi run -e ml "${env_[@]}" spikes/m8/bench/target/release/m8-bench \
    --model ff_models/exports/m8b-off23-small-f64.pt --xyz spikes/m10/out/boxes/nma_w216_m8b.xyz \
    --p $p --chunk 8 --dtype f32 --device cuda --threads 1 --budget-s 30 --max-iter 10 | tee -a "$out/bench.jsonl"
done
