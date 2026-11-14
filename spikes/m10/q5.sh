#!/usr/bin/env bash
# M10 spike Q5: budget numbers for a dynamics study.
# (a) 216 pure waters (out/boxes/water_w216.xyz), MACE-OFF23 small f32 CUDA, the M8 bench at
#     P = 1, 8, 32 (c = 8), as Q1;
# (b) one gas-phase H2O (the M6 geometry), merged pimd-rs NVT runs of n_short and n_long steps
#     (ms per step = Δwall / Δsteps) at P = 1, 32, 64 on CPU (1 and 4 libtorch threads, float64 and
#     float32) and CUDA (float64).
# Output: spikes/m10/out/q5/{bench_water.jsonl, gas.txt, gas/<tag>/}.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=spikes/m10/out/q5
mkdir -p "$out/gas"
{ date -Is; uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$out/gas.txt"
env_=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAYON_NUM_THREADS=4 LIBTORCH_USE_PYTORCH=1)
for p in 1 8 32; do
  pixi run -e ml "${env_[@]}" spikes/m8/bench/target/release/m8-bench \
    --model ff_models/exports/m8b-off23-small-f64.pt --xyz spikes/m10/out/boxes/water_w216.xyz \
    --p $p --chunk $(( p < 8 ? p : 8 )) --dtype f32 --device cuda --threads 1 --budget-s 30 --max-iter 10 \
    >> "$out/bench_water.jsonl"
done
gas() {  # P device dtype threads n_short n_long
  local p=$1 dev=$2 dt=$3 th=$4 ns=$5 nl=$6 tag=P$1_$2_$3_t$4
  for n in $ns $nl; do
    local d=$out/gas/${tag}_n$n; rm -rf "$d"; mkdir -p "$d"
    cat > "$d/config.toml" <<EOF
# M10 spike Q5 (b): one H2O, M6 geometry, MACE-OFF23 small
[system]
n_atoms = 3
masses_amu = [15.999, 1.008, 1.008]
symbols = ["O", "H", "H"]
positions_angstrom = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]

[potential]
kind = "mace"
model = "ff_models/exports/m6c-off23-small-f64.pt"
device = "$dev"
dtype = "$dt"
threads = $th

[sampling]
temperature_kelvin = 300.0
n_beads = $p
dt_fs = 0.5
tau0_fs = 20.0
n_equil = 0
n_steps = $n
stride = 1
seed = 1032

[output]
checkpoint_every = $n
trajectory_every = $n
EOF
    local s e
    s=$(date +%s.%N)
    RAYON_NUM_THREADS=1 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 target/release/pimd-rs "$d/config.toml" --out "$d" > "$d/stdout.log" 2>&1
    e=$(date +%s.%N)
    echo "$tag n$n wall_s $(echo "$e - $s" | bc)" | tee -a "$out/gas.txt"
  done
}
for p in 1 32 64; do
  gas $p cpu float64 1 200 2000
  gas $p cpu float32 1 200 2000
  gas $p cpu float64 4 200 2000
  gas $p cuda float64 1 200 2000
done
