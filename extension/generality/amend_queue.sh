#!/bin/bash
# Amendment 1: after each data set's frozen scoring, recompute its savings with the amended targets.
cd "$(dirname "$0")"
while [ ! -f results/colon_amend1.json ]; do sleep 30; done
for d in stephenson hao scala_m1 bmmc_cite banc gouwens_visp bmmc_multiome malecns banc_crossanimal; do
  while [ ! -f "results/$d.json" ]; do sleep 60; done
  if [ "$d" = "stephenson" ]; then
    python3 with_object_arrays.py grun_amend.py "$d" > "logs/${d}_amend1.log" 2>&1
  else
    python3 grun_amend.py "$d" > "logs/${d}_amend1.log" 2>&1
  fi
done
echo done > logs/amend_queue.done
