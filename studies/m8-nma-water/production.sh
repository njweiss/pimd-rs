#!/usr/bin/env bash
# The M8b production sequence (docs/experiments/m8-nma-water.md, option b), one run after another:
# P = 1 from the box, then the quantum configs from its first and last production frames, then
# P = 8, 16, 32 and the second P = 32. Stops at the first failure. Progress in out/production.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
s=studies/m8-nma-water
log=$s/out/production.log
mkdir -p "$s/out"
stamp() { echo "$(date '+%F %T') $*" | tee -a "$log"; }
run() { stamp "start $1"; "$s/run.sh" "$1" | tee -a "$log"; stamp "end $1"; }
run $s/configs/p1/p1.toml
python3 $s/make_configs.py quantum | tee -a "$log"
for r in p8 p16 p32 p32_seed2; do run $s/configs/quantum/$r.toml; done
stamp "production done"
