#!/bin/bash
# Two queues of the generality benchmark (run after freeze.py); each data set: predict, then evaluate.
cd "$(dirname "$0")"
queue() { for d in "$@"; do python3 grun.py run "$d" > "logs/$d.log" 2>&1; done; }
queue stephenson colon scala_m1 banc banc_crossanimal > /dev/null 2>&1 &
queue hao bmmc_cite bmmc_multiome gouwens_visp malecns > /dev/null 2>&1 &
wait
echo done > logs/queues.done
