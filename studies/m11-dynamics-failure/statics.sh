#!/usr/bin/env bash
# M11 pilot step 3: static <V> and <E> vs P (plain PIMD, PILE-L, 2000 equilibration + 20000 steps).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=studies/m11-dynamics-failure/out/pilot/statics
run() { python3 studies/m11-dynamics-failure/run.py "$out/$1_T$2_P$3_dt$4" --system $1 --t $2 --p $3 \
          --dt-fs $4 --k 4 --static 20000 --seed $5; }
export -f run; export out
{
for s in s1 s2; do
  for p in 16 32 64; do echo "$s 300 $p 0.1 21$p"; done
  for p in 48 96 128; do echo "$s 100 $p 0.1 22$p"; done
  echo "$s 300 32 0.05 2332"
  echo "$s 100 96 0.05 2396"
done
} | xargs -P 3 -L 1 bash -c 'run "$@"' _
