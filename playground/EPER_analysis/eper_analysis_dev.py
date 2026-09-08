from eregion.tasks.ptc import PTCResult
from eregion.tasks.eper import PTCEPERFitter

import matplotlib.pyplot as plt
import logging

FPATH: str = "/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260721-174626"


PTCtable = PTCResult.load(FPATH)


s_settings = {"N_transfers" : 1024, "do_trapfit": False}

p_settings = {"N_transfers" : 4104,
              "subtract_eper_zeros": 5, "do_trapfit": False}


epertask = PTCEPERFitter(pass_columns=["det_id", "output", "exptime"],
                         siglevelcol="mean_0",
                         ser_eper_col="ser_eper_mean_0",
                         llel_eper_col="llel_eper_mean_0",
                         ser_settings=s_settings,
                         llel_settings=p_settings
                         )
epertask.logger.setLevel(logging.DEBUG)

results = epertask.run(PTCtable)


plt.close("all")

fig, ax = plt.subplots(1,2, figsize=(10,8))


for ident, curve in results.eper_table.groupby(["det_id", "output"]):

    lab = f"{ident[0]}_{ident[1]}"
    ax[0].plot(curve["signal_level"], curve["ser_CTI_jan"], ".", label=lab)
    ax[1].plot(curve["signal_level"], curve["llel_CTI_jan"], ".", label=lab)

    

ax[0].legend()
ax[1].legend()

ax[0].set_xlabel("signal level (DN)")
ax[1].set_xlabel("signal level (DN)")

ax[0].set_ylabel("serial CTI")
ax[1].set_ylabel("parallel CTI")

ax[0].loglog()
ax[1].loglog()


fig.tight_layout()    



    

