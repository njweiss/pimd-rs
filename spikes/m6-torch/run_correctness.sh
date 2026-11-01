#!/usr/bin/env bash
# Spike questions 3 and 4: Rust (tch, TorchScript, Rust edges) vs Python MACE (MACECalculator,
# mace's neighbour list). Usage, from the repo root:
#   spikes/m6-torch/run_correctness.sh NAME MODEL_FILE EXPORT_DIR "SYSTEMS"
# e.g. spikes/m6-torch/run_correctness.sh tiny spikes/m6-torch/out/tiny.model spikes/m6-torch/out "water ethanol waterbox"
set -euo pipefail
name=$1 model=$2 xdir=$3 systems=$4
S=spikes/m6-torch
B=$S/target/release/m6-torch-spike
R=$S/out/res/$name
mkdir -p "$R" "$xdir"
export OMP_NUM_THREADS=1
py() { pixi run -e ml python "$@" 2>&1 | grep -vE "Warning|warn\(|visitor|cuequivariance|_Jd|weights_only|torch.load|^$" || true; }
for d in float64 float32; do
  [ -f "$xdir/$name-$d.pt" ] || py $S/export_mace.py "$model" --dtype $d --out "$xdir/$name-$d.pt"
done
for sys in $systems; do
  g=$S/out/geoms/${sys}_p8.txt
  for d in float64 float32; do
    py $S/ref_eval.py "$model" "$g" "$R/py-$sys-$d-cpu.txt" --dtype $d --device cpu --edges "$R/py-$sys-edges.txt"
    py $S/ref_eval.py "$model" "$g" "$R/py-$sys-$d-cuda.txt" --dtype $d --device cuda
  done
  $B edges "$xdir/$name-float64.pt" "$g" "$R/rs-$sys-edges.txt"
  echo "== $name $sys Q4 edge set (Rust vs mace get_neighborhood), 8 beads"
  .pixi/envs/ml/bin/python $S/compare.py edges "$R/rs-$sys-edges.txt" "$R/py-$sys-edges.txt"
  for d in float64 float32; do
    t=$([ $d = float64 ] && echo f64 || echo f32)
    for dev in cpu cuda; do
      $B eval "$xdir/$name-$d.pt" "$g" "$R/rs-$sys-$d-$dev.txt" --device $dev --threads 1 2>/dev/null
      echo -n "Q3 $name $sys $d $dev  Rust vs Python same dtype/device: "
      .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-$d-$dev.txt" "$R/py-$sys-$d-$dev.txt"
    done
  done
  echo -n "Q3 $name $sys float32 vs float64 (Rust, CPU): "
  .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-float32-cpu.txt" "$R/rs-$sys-float64-cpu.txt"
  echo -n "Q3 $name $sys Rust cuda f64 vs Python cpu f64: "
  .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-float64-cuda.txt" "$R/py-$sys-float64-cpu.txt"
  $B eval "$xdir/$name-float64.pt" "$g" "$R/rs-$sys-f64-cast32.txt" --dtype f32 --threads 1 2>/dev/null
  echo -n "Q2 $name $sys f64 export cast to f32 in Rust (CModule::to) vs f32 export: "
  .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-f64-cast32.txt" "$R/rs-$sys-float32-cpu.txt"
  $B eval "$xdir/$name-float64.pt" "$g" "$R/rs-$sys-f64-pyedges.txt" --threads 1 --edges-file "$R/py-$sys-edges.txt" 2>/dev/null
  echo -n "Q4 $name $sys Rust model, mace edges vs Rust edges (f64 cpu): "
  .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-f64-pyedges.txt" "$R/rs-$sys-float64-cpu.txt"
  $B eval "$xdir/$name-float64.pt" "$g" "$R/rs-$sys-f64-loop.txt" --threads 1 --mode loop 2>/dev/null
  echo -n "Q5 $name $sys batched vs per-bead loop (f64 cpu): "
  .pixi/envs/ml/bin/python $S/compare.py results "$R/rs-$sys-f64-loop.txt" "$R/rs-$sys-float64-cpu.txt"
done
