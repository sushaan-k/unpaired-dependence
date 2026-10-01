#!/bin/sh
# Development runs, one data set at a time (memory).
for n in hao stephenson bmmc_cite colon bmmc_multiome scala_m1 gouwens_visp banc malecns; do
  python3 dev.py run $n
done
echo DEV_DONE
