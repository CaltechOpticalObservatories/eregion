import eregion
from eregion.tasks.ptc_fitting import CCDPTCFit, PTCResult
import os
import logging

logging.basicConfig(level=logging.DEBUG)


DATA_DIR="/scratch/DEIMOS/DTU_detreduce/PTC/SCI/20260721-095716/"


res = PTCResult.load(DATA_DIR)

task = CCDPTCFit(selection_columns=["det_id", "output"])


it = task.iter_ptc_curves(res.ptc_table)
