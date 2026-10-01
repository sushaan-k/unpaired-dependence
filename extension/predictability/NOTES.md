# Notes written after the data record (results/data_freeze.json)

`DEVIATIONS.md` is hashed in the data record and is not edited afterwards; later notes go here.

* Clerical (after extraction): Deviation 1 quotes 30,672 cells for bmcite, the number reported with the original
  data. The extracted file has 33,454 cells, every barcode present in both the RNA and the antibody count files.
  Nothing else depends on it.
* The test harness `test/run_test.py` ran `prun.py` on two development data sets (colon, banc) before the freeze to
  check the code; its outputs in `test/results` are not results.
