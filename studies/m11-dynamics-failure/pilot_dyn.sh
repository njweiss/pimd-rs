#!/usr/bin/env bash
# M11 pilot step 4: 10 segments per method at 300 K (P = 64; classical P = 1), both systems.
set -euo pipefail
cd "$(dirname "$0")/../.."
export out=studies/m11-dynamics-failure/out/pilot/dyn
run() { python3 studies/m11-dynamics-failure/run.py "$out/$1_T300_$2" --system $1 --t 300 \
          --method $2 --p $3 --k $4 --segments 10 --seed $5; }
export -f run
{
for s in s1 s2; do
  echo "$s rpmd 64 3 41${s:1}1"; echo "$s trpmd 64 3 41${s:1}2"; echo "$s pa-cmd 64 3 41${s:1}3"
  echo "$s classical 1 1 41${s:1}4"
done
} | xargs -P 3 -L 1 bash -c 'run "$@"' _
