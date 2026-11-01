#!/usr/bin/env bash
# Spike question 5: time per evaluation (all beads; includes Rust edge building, host<->device
# copies and the copy back), batched vs a per-bead loop, CPU 1 / 4 libtorch threads (OMP_NUM_THREADS
# set to the same value), CUDA, float64 / float32, TorchScript profiling executor off (the mode Q6
# recommends). Median of REPS calls after WARMUP calls. From the repo root:
#   spikes/m6-torch/run_bench.sh > spikes/m6-torch/out/q5.log
set -uo pipefail
S=spikes/m6-torch
B=$S/target/release/m6-torch-spike
G=$S/out/geoms
declare -A X=([tiny]=$S/out/tiny-float64.pt [off23-small]=ff_models/exports/off23-small-float64.pt
              [off23-medium]=ff_models/exports/off23-medium-float64.pt)
echo "# start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
for m in tiny off23-small off23-medium; do
  if [ $m = off23-medium ]; then reps=5 warm=2; else reps=10 warm=3; fi
  for sys in water ethanol; do
    for p in 1 8 16 32; do
      for dt in f64 f32; do
        for dev in cpu1 cpu4 cuda; do
          case $dev in cpu1) d="--device cpu --threads 1"; t=1;; cpu4) d="--device cpu --threads 4"; t=4;;
                       cuda) d="--device cuda --threads 1"; t=1;; esac
          modes="batched"
          if [ $p -gt 1 ] && { [ $m != off23-medium ] || [ $p = 8 ]; }; then modes="batched loop"; fi
          for mode in $modes; do
            r=$(OMP_NUM_THREADS=$t $B bench ${X[$m]} $G/${sys}_p$p.txt $d --dtype $dt --mode $mode \
                --no-profiling --reps $reps --warmup $warm 2>&1 | sed 's/.*| //')
            echo "$m $sys P=$p $dt $dev $mode | $r | load $(cut -d' ' -f1 /proc/loadavg)"
          done
        done
      done
    done
  done
done
echo "# end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
