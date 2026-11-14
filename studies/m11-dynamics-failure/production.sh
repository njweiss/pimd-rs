#!/usr/bin/env bash
# M11 tier 1 production: 4 replicas x 10 segments per (system, T, method); the pilot's 300 K runs
# (out/pilot/dyn/, same settings) are replica 0 at 300 K. Then the Q4 convergence runs (2 replicas).
# Three runs at a time, K = 3 clients each (classical: K = 1). Jobs: system T method P K dt omega seed tag
set -euo pipefail
cd "$(dirname "$0")/../.."
export out=studies/m11-dynamics-failure/out/prod
run() {
  local d="$out/$9"
  if [ -f "$d/correlations.csv" ]; then echo "$d done"; return 0; fi
  python3 studies/m11-dynamics-failure/run.py "$d" --system $1 --t $2 --method $3 --p $4 --k $5 \
    --dt-fs $6 --omega-cm $7 --segments 10 --seed $8
}
export -f run
jobs() {
  local s m r
  # 300 K production, replicas 1-3
  for r in 1 2 3; do for s in s1 s2; do
    for m in rpmd trpmd pa-cmd; do echo "$s 300 $m 64 3 0.1 16000 5${s:1}3${r}$(( ${#m} )) ${s}_T300_${m}_r$r"; done
    echo "$s 300 classical 1 1 0.1 16000 5${s:1}3${r}0 ${s}_T300_classical_r$r"
  done; done
  # Q4 convergence at 300 K, 2 replicas
  for r in 0 1; do
    for s in s1 s2; do
      echo "$s 300 rpmd 128 3 0.1 16000 6${s:1}1${r} ${s}_T300_rpmd_P128_r$r"
      echo "$s 300 pa-cmd 128 3 0.1 16000 6${s:1}2${r} ${s}_T300_pa-cmd_P128_r$r"
      echo "$s 300 rpmd 64 3 0.05 16000 6${s:1}3${r} ${s}_T300_rpmd_dt0.05_r$r"
      echo "$s 300 pa-cmd 64 3 0.05 32000 6${s:1}4${r} ${s}_T300_pa-cmd_W32000_dt0.05_r$r"
    done
  done
  # 100 K production, replicas 0-3
  for r in 0 1 2 3; do for s in s1 s2; do
    for m in rpmd trpmd pa-cmd; do echo "$s 100 $m 192 3 0.1 16000 7${s:1}1${r}$(( ${#m} )) ${s}_T100_${m}_r$r"; done
    echo "$s 100 classical 1 1 0.1 16000 7${s:1}1${r}0 ${s}_T100_classical_r$r"
  done; done
}
jobs | xargs -P 3 -L 1 bash -c 'run "$@"' _
