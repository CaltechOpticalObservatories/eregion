import sys


#I know I know, no idea why this happens, can't import eregion if I don't do this
sys.path.append("/home/danw/Software/eregion")
sys.path.append("/home/danw/Software/eregion/eregion")


import eregion
from eregion.tasks.ptc_fitting import CCDPTCFit, PTCResult, BrighterFatterFitTypes
import os
import logging


DATA_DIR = "/scratch/DEIMOS/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260720-101920"


ptc_results = PTCResult.load(DATA_DIR)

task = CCDPTCFit(selection_columns=["det_id", "output"], brighter_fatter=BrighterFatterFitTypes.ASTIER_ONE_PARAM)
task.logger.setLevel(logging.DEBUG)

res = task.run(ptc_results)
