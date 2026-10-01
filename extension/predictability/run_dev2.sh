#!/bin/sh
for n in hao stephenson bmmc_cite colon bmmc_multiome scala_m1 gouwens_visp banc malecns; do
  python3 dev2.py $n
done
echo DEV2_DONE
