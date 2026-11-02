#!/usr/bin/env bash
# Run one M6c config (docs/experiments/m6-water-mace.md) from the repository root:
#   studies/m6-water-mace/run.sh studies/m6-water-mace/configs/<set>/<run>.toml
# Output: studies/m6-water-mace/out/<set>/<run>/ (observables.csv, trajectory.xyz, run.log,
# config.toml as run, wall.txt). xTB configs (address @ADDRESS@) get four single-threaded
# ipi_client.py xtb clients from the pixi `ref` env, as in M5. The engine is the release build of
# pimd-cli with feature `torch`, run inside the pixi `ml` env.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=$1
name=$(basename "$cfg" .toml)
set_=$(basename "$(dirname "$cfg")")
out=studies/m6-water-mace/out/$set_/$name
mkdir -p "$out"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
if grep -q '@ADDRESS@' "$cfg"; then
  sock=$(mktemp -u /tmp/m6c-XXXXXX.sock)
  sed "s|@ADDRESS@|unix:$sock|" "$cfg" > "$out/config.toml"
  for _ in 1 2 3 4; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      .pixi/envs/ref/bin/python crates/pimd-socket/tests/clients/ipi_client.py \
      --address "unix:$sock" xtb --symbols O H H < /dev/null > "$out/client.log" 2>&1 &
    pids+=($!)
  done
else
  cp "$cfg" "$out/config.toml"
fi
start=$(date +%s.%N)
RAYON_NUM_THREADS=1 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 \
  target/release/pimd-rs "$out/config.toml" --out "$out" > "$out/stdout.log" 2>&1
end=$(date +%s.%N)
printf 'wall_s %s\ncommit %s\n' "$(echo "$end - $start" | bc)" "$(git rev-parse --short HEAD)" > "$out/wall.txt"
cat "$out/wall.txt"
