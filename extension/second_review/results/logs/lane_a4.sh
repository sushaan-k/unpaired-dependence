#!/bin/bash
# The validation review: even and odd folds side by side, each with two BLAS threads (as the validation test ran)
# that sleep as soon as they are idle, so that the two runs do not spin against each other.
cd "$(dirname "$0")"
L=results/logs
export OPENBLAS_THREAD_TIMEOUT=4
python3 validation_review.py run --part 0 >> $L/validation_part0.log 2>&1 &
sleep 120
python3 validation_review.py run --part 1 >> $L/validation_part1.log 2>&1 &
wait
touch $L/lane_a.done
