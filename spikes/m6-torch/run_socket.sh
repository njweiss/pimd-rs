#!/usr/bin/env bash
# Spike question 7: the merged engine (target/release/pimd-rs, unchanged) with kind = "socket" and
# four single-threaded Python MACE clients (mace_client.py), water, P = 16, 300 K, dt 0.5 fs, the
# M5 start geometry and protocol. Time per step = (t(N2) - t(N1)) / (N2 - N1), which removes
# start-up (client model loading, connecting). Usage, from the repo root:
#   spikes/m6-torch/run_socket.sh NAME MODEL_FILE [DEVICE] [N1] [N2]
set -euo pipefail
name=$1 model=$2 device=${3:-cpu} n1=${4:-20} n2=${5:-120}
S=spikes/m6-torch
O=$S/out/socket/$name-$device
PY=.pixi/envs/ml/bin/python
export PATH="$PWD/.pixi/envs/ml/bin:$PATH"  # ptxas: torch 2.13 eager CUDA JIT-compiles Triton kernels
rm -rf "$O"; mkdir -p "$O"
run() {  # steps dumpdir
  local steps=$1 dump=$2 sock="$O/w.sock" pids=()
  cat > "$O/water_$steps.toml" <<TOML
[system]
n_atoms = 3
masses_amu = [15.999, 1.008, 1.008]
symbols = ["O", "H", "H"]
positions_angstrom = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]

[potential]
kind = "socket"
address = "unix:$PWD/$sock"
n_clients = 4
timeout_s = 600.0

[sampling]
temperature_kelvin = 300.0
n_beads = 16
dt_fs = 0.5
tau0_fs = 20.0
n_equil = 0
n_steps = $steps
stride = 1
seed = 31337

[output]
checkpoint_every = $steps
trajectory_every = $steps
TOML
  for c in 1 2 3 4; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $S/mace_client.py \
      --address "unix:$PWD/$sock" "$model" --symbols O H H --device "$device" \
      ${dump:+--dump "$dump" --dump-count ${DUMP_COUNT:-4}} > "$O/client_${steps}_$c.log" 2>&1 &
    pids+=($!)
  done
  local t0=$(date +%s.%N)
  ./target/release/pimd-rs "$O/water_$steps.toml" --out "$O/run_$steps" > "$O/engine_$steps.log" 2>&1
  local t1=$(date +%s.%N)
  wait "${pids[@]}" || true
  echo "$steps $(echo "$t1 - $t0" | bc)"
}
mkdir -p "$O/dump"
uptime
a=$(run "$n1" "$O/dump"); echo "run $a s"
b=$(run "$n2" ""); echo "run $b s"
echo "$a $b" | awk '{printf "Q7 per step: %.2f ms (P = 16, 4 clients)\n", 1000*($4-$2)/($3-$1)}'
uptime
