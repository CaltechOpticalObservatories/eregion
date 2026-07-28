import sys
import pathlib
from utils.pydantic import dataframe_from_tabular_model


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


tabresults = []
dirname = []
dfs = []

for root, dirs, files in TL_DATA_DIR.walk():
    if not root.match("*/old/*"):
        if "ptc_table.fits" in files:
            print(f"fitting PTCs in directory {root}")
            ptc_results = PTCResult.load(root)
#            task = CCDPTCFit(selection_columns=["det_id", "output"], brighter_fatter=BrighterFatterFitTypes.ASTIER_ONE_PARAM, fwfact=0.9)
            task = CCDPTCFitTabular(selection_columns=["det_id", "output"], brighter_fatter=BrighterFatterFitTypes.ASTIER_ONE_PARAM)
            task.logger.setLevel(logging.DEBUG)
            result = task.run(ptc_results)

            outdir = OUT_DATA_DIR / root.name

            tabresults.append(result)
            dirname.append(root.name)

            df = dataframe_from_tabular_model(result)

            df["flux_rate_cal"] = df["flux_rate"] * df["camera_gain_BFE"]
            df["full_well_cal"] = df["full_well"] * df["camera_gain_BFE"]
            mm = max(df["flux_rate"])
            relQE = [ (_ / mm).nominal_value for _  in df["flux_rate"]]
            df["relQE"]  = relQE

            dfs.append(df)
#            result.save(outdir)



QEind = dirname.index("20260721-174626")
QEtbl = dfs[QEind][["output_ids", "flux_rate", "relQE"]].sort_values(by="relQE")
ordering = [f"{a[0]}[{a[1]}]" for a in QEtbl["output_ids"].array]
print("\n".join(ordering))
