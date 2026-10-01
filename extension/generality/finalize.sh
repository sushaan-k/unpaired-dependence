#!/bin/bash
# After every data set is scored: remaining amendment runs, the law's predictions, summaries and assets, one at a
# time (memory; see README).
cd "$(dirname "$0")"
while [ ! -f logs/run_rest.done ] || [ -n "$(ps -eo cmd | grep 'grun.py run bmmc_multiome' | grep -v grep)" ]; do sleep 30; done
for d in gouwens_visp bmmc_multiome malecns banc_crossanimal; do python3 grun_amend.py "$d" > "logs/${d}_amend1.log" 2>&1; done
python3 with_object_arrays.py grun_amend.py stephenson > logs/stephenson_amend1.log 2>&1
python3 with_object_arrays.py glaw.py predict > logs/glaw_predict.log 2>&1
python3 grun.py summary > logs/summary.log 2>&1
python3 glaw.py test > logs/glaw_test.log 2>&1
python3 grun_amend.py summary > logs/summary_amend1.log 2>&1
python3 grun_amend.py law > logs/law_amend1.log 2>&1
python3 make_assets.py > logs/make_assets.log 2>&1
echo done > logs/finalize.done
