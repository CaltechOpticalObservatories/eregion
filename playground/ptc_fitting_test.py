import sys
import pathlib

#I know I know, no idea why this happens, can't import eregion if I don't do this
sys.path.append("/home/danw/Software/eregion")
sys.path.append("/home/danw/Software/eregion/eregion")


import eregion
from eregion.tasks.ptc_fitting import CCDPTCFit, PTCResult, BrighterFatterFitTypes, CCDPTCFitTabular
import os
import logging


TL_DATA_DIR = pathlib.Path("/scratch/DEIMOS/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/")
OUT_DATA_DIR = pathlib.Path("./ptc_fits/")
OUT_DATA_DIR.mkdir(exist_ok=True, parents=True)


for root, dirs, files in TL_DATA_DIR.walk():
    if not root.match("*/old/*"):
        if "ptc_table.fits" in files:
            print(f"fitting PTCs in directory {root}")
            ptc_results = PTCResult.load(root)
#            task = CCDPTCFit(selection_columns=["det_id", "output"], brighter_fatter=BrighterFatterFitTypes.ASTIER_ONE_PARAM)
            task = CCDPTCFitTabular(selection_columns=["det_id", "output"], brighter_fatter=BrighterFatterFitTypes.ASTIER_ONE_PARAM)
            task.logger.setLevel(logging.DEBUG)
            result = task.run(ptc_results)

            outdir = OUT_DATA_DIR / root.name
            #result.save(outdir)
