from eregion.tasks.ptc import PTCResult
from eregion.tasks.eper import PTCEPERFitter

import matplotlib.pyplot as plt
import logging

FPATH: str = "/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260721-174626"


PTCtable = PTCResult.load(FPATH)


s_settings = {"N_transfers" : 1024}

p_settings = {"N_transfers" : 4104,
              "subtract_eper_zeros": 5}


epertask = PTCEPERFitter(selection_columns=["det_id", "output"],
                         fluxcolname="exptime",
                         siglevelcolname="mean_0",
                         ser_eper_colname="ser_eper_mean_0",
                         llel_eper_colname="llel_eper_mean_0",
                         ser_settings=s_settings,
                         llel_settings=p_settings
                         )
epertask.logger.setLevel(logging.DEBUG)


plt.close("all")


    
