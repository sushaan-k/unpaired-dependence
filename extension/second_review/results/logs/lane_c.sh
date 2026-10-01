#!/bin/bash
# The jobs that need more memory than is free while both validation parts run (the atlas bootstrap was stopped by
# the memory limit at 07:06, and stephenson.npz needs the benchmark's reader for string arrays, its Deviation 1).
# Part 1 of the validation review is stopped between two folds and restarted afterwards; it skips finished folds.
cd "$(dirname "$0")"
L=results/logs
export OPENBLAS_THREAD_TIMEOUT=4
P0=5247
P1=5282
while [ ! -f $L/lane_b.done ]; do sleep 30; done
if kill -0 $P1 2>/dev/null; then
    n0=$(grep -c " done \[" $L/validation_part1.log)
    while kill -0 $P1 2>/dev/null && [ "$(grep -c " done \[" $L/validation_part1.log)" -eq "$n0" ]; do sleep 20; done
    sleep 60
    f=$(ls -t results/validation/fold*.npz | head -1)
    python3 -c "import numpy as np, sys; assert len(np.load(sys.argv[1]).files) > 0" "$f" && kill $P1
    echo "[$(date +%H:%M:%S)] part 1 stopped after $f" >> $L/lane_c.out
fi
python3 ../generality/with_object_arrays.py benchmark_review.py run stephenson > $L/benchmark_stephenson.log 2>&1
echo "[$(date +%H:%M:%S)] benchmark_review stephenson exit $?" >> $L/lane_c.out
(cd ../checks && python3 ../generality/with_object_arrays.py draw_scoring.py benchmark stephenson --folds 0 \
    > results/log_benchmark_stephenson.txt 2>&1)
echo "[$(date +%H:%M:%S)] draw_scoring stephenson exit $?" >> $L/lane_c.out
# part 1 resumes as soon as part 0 has finished its folds or the atlas bootstrap is done, whichever comes first
( while kill -0 $P0 2>/dev/null && [ ! -f $L/atlas_bootstrap.done ]; do sleep 30; done
  echo "[$(date +%H:%M:%S)] part 1 restarted" >> $L/lane_c.out
  python3 validation_review.py run --part 1 >> $L/validation_part1.log 2>&1
  echo "[$(date +%H:%M:%S)] part 1 exit $?" >> $L/lane_c.out ) &
python3 atlas_bootstrap.py run -1 0 1 2 3 4 5 6 7 8 9 > $L/atlas_bootstrap.log 2>&1
echo "[$(date +%H:%M:%S)] atlas_bootstrap exit $?" >> $L/lane_c.out
touch $L/atlas_bootstrap.done
wait
touch $L/lane_c.done
