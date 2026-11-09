#!/usr/bin/env bash
# Run the named quantum runs of the M8b production sequence in order (docs/experiments/
# m8-nma-water.md), e.g. `resume.sh p32 p32_seed2`: a run with a checkpoint restarts from it, any
# other starts fresh. health.py runs after each run; the sequence stops at the first failed run or
# health check. Progress in out/production.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
s=studies/m8-nma-water
log=$s/out/production.log
stamp() { echo "$(date '+%F %T') $*" | tee -a "$log"; }
for r in "$@"; do
  restart=""
  [ ! -e "$s/out/quantum/$r/checkpoint.bin" ] || restart=--restart
  stamp "start $s/configs/quantum/$r.toml $restart"
  "$s/run.sh" "$s/configs/quantum/$r.toml" $restart | tee -a "$log"
  stamp "end $s/configs/quantum/$r.toml"
  nice -n 19 pixi run -e ref python $s/health.py $s/out/quantum/$r | tee -a "$log"
done
stamp "production done"
