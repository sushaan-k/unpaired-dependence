#!/usr/bin/env python3
"""Deviation 1 (DEVIATIONS.md): run a frozen script with numpy.load allowed to read the object (string) arrays that
the extraction left in stephenson.npz. No frozen file and no data file is changed; values are unaffected.

    python with_object_arrays.py grun.py run stephenson
    python with_object_arrays.py glaw.py predict
"""

import runpy
import sys

import numpy as np

_load = np.load


def load(path, *args, **kwargs):
    if str(path).endswith("stephenson.npz"):
        kwargs["allow_pickle"] = True
    return _load(path, *args, **kwargs)


np.load = load
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
