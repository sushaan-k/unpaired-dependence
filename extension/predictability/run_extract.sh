#!/bin/sh
for n in pbmc10k malt10k human_gaba snare_cortex sln111 sln206 bmcite fafb_vpn banc_vpn fetal_cortex; do
  echo "== $n $(date +%T)"; python3 pextract.py $n
done
echo EXTRACT_DONE
