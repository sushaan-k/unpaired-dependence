#!/bin/bash
# Continues lane_c.sh, whose benchmark_review.py run for stephenson failed through the reader wrapper because the
# script's own folder was not on the module path (the wrapper runs it with runpy): it waits for the draw-scoring
# check of stephenson, runs benchmark_review.py for stephenson with this folder on PYTHONPATH, then starts the atlas
# bootstrap and restarts part 1 of the validation review as lane_c.sh would have.
cd "$(dirname "$0")"
L=results/logs
export OPENBLAS_THREAD_TIMEOUT=4
P0=5247
DS=11126
note() { echo "[$(date +%H:%M:%S)] $*" >> $L/lane_c.out; }
while kill -0 $DS 2>/dev/null; do sleep 20; done
note "draw_scoring stephenson finished (see ../checks/results/log_benchmark_stephenson.txt)"
PYTHONPATH="$PWD" python3 ../generality/with_object_arrays.py benchmark_review.py run stephenson \
    > $L/benchmark_stephenson.log 2>&1
s=$?
note "benchmark_review stephenson exit $s"
( while kill -0 $P0 2>/dev/null && [ ! -f $L/atlas_bootstrap.done ]; do sleep 30; done
  note "part 1 restarted"
  python3 validation_review.py run --part 1 >> $L/validation_part1.log 2>&1
  s=$?
  note "part 1 exit $s" ) &
python3 atlas_bootstrap.py run -1 0 1 2 3 4 5 6 7 8 9 > $L/atlas_bootstrap.log 2>&1
s=$?
note "atlas_bootstrap exit $s"
touch $L/atlas_bootstrap.done
wait
touch $L/lane_c.done
