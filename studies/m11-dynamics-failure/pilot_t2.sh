#!/usr/bin/env bash
# M11 tier 2 pilot steps 3-4 (docs/experiments/m11-tier2-water.md), one GPU job at a time.
# Batched client; route "cueq" = cueq f32 (.venv-cueq), "f64" = e3nn f64 (pixi ml).
set -uo pipefail
cd "$(dirname "$0")/../.."
O=studies/m11-dynamics-failure/out/tier2/pilot
CUEQ=(--python "pixi run -e ml .venv-cueq/bin/python" --client-args "--dtype float32 --backend cueq")
F64=(--client-args "--dtype float64 --backend e3nn")
run() {  # dir, then run_water.py arguments
  local d="$O/$1"; shift
  if [ -f "$d/wall.txt" ] && grep -q 'rc 0' "$d/wall.txt"; then echo "$d done"; return 0; fi
  python3 studies/m11-dynamics-failure/run_water.py "$d" --mux "$@"
}
# 3a. Precision: classical P = 1, N = 64, 5 segments; e3nn f64 against cueq f32 (same seed).
run prec/classical_cueq --t 300 --p 1 --method classical --n-mol 64 --segments 5 --seed 2301 "${CUEQ[@]}"
run prec/classical_f64  --t 300 --p 1 --method classical --n-mol 64 --segments 5 --seed 2301 "${F64[@]}"
# 3b. Precision: RPMD P = 64, N = 4, 1 segment (NVE drift from segments.csv).
run prec/rpmd_cueq --t 300 --p 64 --method rpmd --n-mol 4 --segments 1 --seed 2302 "${CUEQ[@]}"
run prec/rpmd_f64  --t 300 --p 64 --method rpmd --n-mol 4 --segments 1 --seed 2302 "${F64[@]}"
# 3c. Statics: P scan, N = 64, 1000 + 5000 steps of plain PIMD.
for p in 16 32 64 128; do run statics/T300_P$p --t 300 --p $p --n-mol 64 --static 5000 --n-equil 1000 --seed 2310 "${CUEQ[@]}"; done
for p in 32 64 128 256; do run statics/T150_P$p --t 150 --p $p --n-mol 64 --static 5000 --n-equil 1000 --seed 2311 "${CUEQ[@]}"; done
# 3d. PA-CMD at 150 K, P = 128, Δt = 0.05 fs: ⟨V⟩ and stability (short segments).
run pacmd150 --t 150 --p 128 --method pa-cmd --n-mol 64 --dt-fs 0.05 --n-equil 4000 --decorr 2000 \
  --seg-steps 4000 --stride 10 --n-lags 401 --segments 1 --seed 2320 "${CUEQ[@]}"
# 4. One short run per method at 300 K, P = 64, N = 64, 3 segments (the RPMD run is also the Δt
# reference); then RPMD at Δt = 0.125 fs on the same lag grid (3e, the Δt check).
run dyn/classical --t 300 --p 1 --method classical --n-mol 64 --segments 3 --seed 2331 "${CUEQ[@]}"
run dyn/rpmd --t 300 --p 64 --method rpmd --n-mol 64 --segments 3 --seed 2332 "${CUEQ[@]}"
run dyn/trpmd --t 300 --p 64 --method trpmd --n-mol 64 --segments 3 --seed 2333 "${CUEQ[@]}"
run dyn/pa-cmd --t 300 --p 64 --method pa-cmd --n-mol 64 --dt-fs 0.1 --n-equil 2000 --decorr 1000 \
  --seg-steps 20000 --stride 5 --segments 3 --seed 2334 "${CUEQ[@]}"
run dyn/rpmd_dt0.125 --t 300 --p 64 --method rpmd --n-mol 64 --dt-fs 0.125 --n-equil 1600 \
  --decorr 800 --seg-steps 16000 --stride 4 --segments 3 --seed 2335 "${CUEQ[@]}"
echo "pilot_t2.sh finished"
