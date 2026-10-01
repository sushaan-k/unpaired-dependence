#!/bin/sh
# Prospective test: all predictions first (hashed), then all evaluations, then the summary.
for n in pbmc10k malt10k human_gaba fafb_vpn banc_vpn snare_cortex fetal_cortex sln111 sln206 bmcite; do
  python3 prun.py predict $n || echo "PREDICT_FAILED $n"
done
echo PREDICT_ALL_DONE
for n in pbmc10k malt10k human_gaba fafb_vpn banc_vpn snare_cortex fetal_cortex sln111 sln206 bmcite; do
  python3 prun.py evaluate $n || echo "EVALUATE_FAILED $n"
done
python3 prun.py summary
echo PROSPECTIVE_DONE
