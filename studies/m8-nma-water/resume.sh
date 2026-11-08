#!/usr/bin/env bash
# Resume the M8b production sequence after the stop of 2026-11-08 (docs/experiments/m8-nma-water.md):
# P = 16 from its checkpoint, then P = 32 and the second P = 32. health.py runs after each run;
# the sequence stops at the first failed run or health check. Progress in out/production.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
s=studies/m8-nma-water
log=$s/out/production.log
stamp() { echo "$(date '+%F %T') $*" | tee -a "$log"; }
run() {
  stamp "start $1 $2"
  "$s/run.sh" "$1" $2 | tee -a "$log"
  stamp "end $1"
  r=$(basename "$1" .toml)
  nice -n 19 pixi run -e ref python $s/health.py $s/out/quantum/$r | tee -a "$log"
}
run $s/configs/quantum/p16.toml --restart
for r in p32 p32_seed2; do run $s/configs/quantum/$r.toml ""; done
stamp "production done"
