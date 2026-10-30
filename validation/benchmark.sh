#!/usr/bin/env bash
# Performance table for validation/RESULTS.md (MILESTONE_4 "README/benchmark deliverables").
#
#   ./validation/benchmark.sh            (from the repository root; needs the pixi `ref` env)
#   REPEATS=5 ./validation/benchmark.sh  (best of REPEATS timings per entry, default 3)
#
# System: the liquid-neon state point of validation/configs/neon_p*.toml (N = 108, 14.1 Å box,
# LJ ε = 35.6 K, σ = 2.749 Å, r_c = 2.2σ shifted, skin 0.3σ, T = 30 K, dt = 2 fs, Langevin
# τ = 100 fs), started from the simple cubic lattice and equilibrated for 1 000 steps.
#
# pimd-rs: the release binary, PILE-L, no RDF; time per step = (wall(1 000 equil + 10 + S
# production) − wall(1 000 equil + 10 production)) / S, so start-up, equilibration and the final
# checkpoint cancel. LAMMPS: `fix nve` + `fix langevin`, neighbor list rebuilt on the same skin
# with `check yes`; time per step from the "Loop time" of a run of S steps after 1 000
# equilibration steps. Threads: RAYON_NUM_THREADS for pimd-rs; for LAMMPS the OPENMP package
# (`-sf omp -pk omp T`, one MPI rank) when `lmp -h` lists it, otherwise T MPI ranks.
# atom-steps/s = N · P · steps / s (bead-atom steps for P > 1). Outputs: stdout and
# validation/runs/benchmark/ (gitignored).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REPEATS="${REPEATS:-3}"
THREADS=(1 2 4 8)
N=108
OUT=validation/runs/benchmark
BIN=target/release/pimd-rs
rm -rf "$OUT"
mkdir -p "$OUT"
cargo build --release -p pimd-cli >/dev/null 2>&1

# --- machine and versions --------------------------------------------------------------------
LMP_HELP="$(pixi run -e ref lmp -h 2>/dev/null)"
if grep -A20 '^Installed packages' <<<"$LMP_HELP" | grep -qw OPENMP; then LMP_MODE=openmp; else LMP_MODE=mpi; fi
echo "## Machine"
echo
echo "- CPU: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) logical CPUs ($(lscpu | sed -n 's/^Core(s) per socket: *//p') cores/socket, $(lscpu | sed -n 's/^Thread(s) per core: *//p') threads/core)"
echo "- OS: $(uname -sr)"
echo "- Load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "- pimd-rs: $($BIN --version), $(rustc --version)"
echo "- LAMMPS: $(sed -n 's/^Large-scale Atomic\/Molecular Massively Parallel Simulator - \(.*\)$/\1/p' <<<"$LMP_HELP" | head -1) (conda: $(pixi list -e ref 2>/dev/null | awk '$1=="lammps"{print $2, $3}')), threads via $LMP_MODE"
echo

# --- pimd-rs ---------------------------------------------------------------------------------
pimd_cfg() { # beads n_steps
  sed -e "s/^n_beads = .*/n_beads = $1/" -e "s/^n_equil = .*/n_equil = 1000/" \
      -e "s/^n_steps = .*/n_steps = $2/" -e "s/^checkpoint_every = .*/checkpoint_every = 100000000/" \
      -e "/^rdf_/d" validation/configs/neon_p16.toml
}
wall() { # threads config  -> seconds (best of REPEATS)
  local best=""
  for _ in $(seq "$REPEATS"); do
    local t0 t1 d
    t0=$(date +%s.%N)
    RAYON_NUM_THREADS="$1" "$BIN" "$2" --out "$OUT/scratch" >/dev/null
    t1=$(date +%s.%N)
    d=$(echo "$t1 - $t0" | bc -l)
    if [ -z "$best" ] || [ "$(echo "$d < $best" | bc -l)" = 1 ]; then best=$d; fi
  done
  echo "$best"
}
pimd_rate() { # beads steps threads -> atom-steps/s
  pimd_cfg "$1" 10 >"$OUT/p$1_short.toml"
  pimd_cfg "$1" $((10 + $2)) >"$OUT/p$1_long.toml"
  local a b
  a=$(wall "$3" "$OUT/p$1_short.toml")
  b=$(wall "$3" "$OUT/p$1_long.toml")
  echo "$N * $1 * $2 / ($b - $a)" | bc -l
}

# --- LAMMPS ----------------------------------------------------------------------------------
# Data file: the pimd-rs initial lattice (5 × 5 × 5 sites, spacing 14.1/5 Å, offset ½, i_x fastest, first 108).
{
  echo "neon lattice (validation/benchmark.sh)"
  echo
  echo "$N atoms"
  echo "1 atom types"
  echo "0 14.1 xlo xhi"
  echo "0 14.1 ylo yhi"
  echo "0 14.1 zlo zhi"
  echo
  echo "Masses"
  echo
  echo "1 20.1797"
  echo
  echo "Atoms # atomic"
  echo
  awk -v n=$N 'BEGIN { a = 14.1 / 5; id = 0
    for (z = 0; z < 5; z++) for (y = 0; y < 5; y++) for (x = 0; x < 5; x++)
      if (id < n) { id++; printf "%d 1 %.10f %.10f %.10f\n", id, (x + 0.5) * a, (y + 0.5) * a, (z + 0.5) * a } }'
} >"$OUT/neon.data"
lmp_input() { # steps
  cat <<EOF
units metal
atom_style atomic
boundary p p p
read_data $OUT/neon.data
variable eps equal 35.6*8.617333262e-5
variable sig equal 2.749
pair_style lj/cut \$(2.2*v_sig)
pair_coeff 1 1 \${eps} \${sig}
pair_modify shift yes
neighbor \$(0.3*v_sig) bin
neigh_modify every 1 delay 0 check yes
timestep 0.002
velocity all create 30.0 2026 dist gaussian
fix 1 all nve
fix 2 all langevin 30.0 30.0 0.1 2026
thermo 1000000
run 1000
run $1
EOF
}
lmp_rate() { # steps threads -> atom-steps/s
  lmp_input "$1" >"$OUT/in.bench"
  local best="" t
  for _ in $(seq "$REPEATS"); do
    if [ "$LMP_MODE" = openmp ]; then
      OMP_NUM_THREADS="$2" pixi run -e ref lmp -sf omp -pk omp "$2" -in "$OUT/in.bench" -log "$OUT/log.lammps" -screen none 2>>"$OUT/lammps.stderr"
    else
      OMP_NUM_THREADS=1 pixi run -e ref mpirun -np "$2" lmp -in "$OUT/in.bench" -log "$OUT/log.lammps" -screen none 2>>"$OUT/lammps.stderr"
    fi
    t=$(awk '/^Loop time of/ {x = $4} END {print x}' "$OUT/log.lammps")
    if [ -z "$best" ] || [ "$(echo "$t < $best" | bc -l)" = 1 ]; then best=$t; fi
  done
  echo "$N * $1 / $best" | bc -l
}

# --- table -----------------------------------------------------------------------------------
declare -A R
for t in "${THREADS[@]}"; do
  R[p16_$t]=$(pimd_rate 16 2000 "$t")
  R[p1_$t]=$(pimd_rate 1 20000 "$t")
  R[lmp_$t]=$(lmp_rate 100000 "$t")
done
fmt() { printf "%.3g" "$1"; }
ratio() { printf "%.2f" "$(echo "$1 / $2" | bc -l)"; }
echo "## Atom-steps per second (best of $REPEATS)"
echo
echo "| threads | pimd-rs P = 16 | speedup | pimd-rs P = 1 | speedup | LAMMPS ($LMP_MODE) | speedup | pimd-rs P = 1 / LAMMPS |"
echo "|---|---|---|---|---|---|---|---|"
for t in "${THREADS[@]}"; do
  echo "| $t | $(fmt "${R[p16_$t]}") | $(ratio "${R[p16_$t]}" "${R[p16_1]}") | $(fmt "${R[p1_$t]}") | $(ratio "${R[p1_$t]}" "${R[p1_1]}") | $(fmt "${R[lmp_$t]}") | $(ratio "${R[lmp_$t]}" "${R[lmp_1]}") | $(ratio "${R[p1_$t]}" "${R[lmp_$t]}") |"
done
echo
s4=$(ratio "${R[p16_4]}" "${R[p16_1]}")
if [ "$(echo "$s4 >= 2" | bc -l)" = 1 ]; then verdict=met; else verdict="not met"; fi
echo "4 vs 1 thread at N = 108, P = 16: ${s4}× (target ≥ 2×, reported, not gated: $verdict)."
echo "Load average at end: $(cut -d' ' -f1-3 /proc/loadavg)"
rm -rf "$OUT/scratch"
