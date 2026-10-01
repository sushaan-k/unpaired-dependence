#!/bin/bash
# Queue B was stopped after bmmc_multiome started so that gouwens_visp and malecns run in parallel with it.
cd "$(dirname "$0")"
while [ -n "$(ps -eo cmd | grep 'grun.py run banc_crossanimal' | grep -v grep)" ]; do sleep 30; done
for d in gouwens_visp malecns; do python3 grun.py run "$d" > "logs/$d.log" 2>&1; done
echo done > logs/run_rest.done
