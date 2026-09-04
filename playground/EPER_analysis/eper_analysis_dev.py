from eregion.tasks.ptc import PTCResult
import matplotlib.pyplot as plt


FPATH: str = "/home/danw/dettest_data/DTU_detreduce/PTC/SCI/20260721-174626"


PTCtable = PTCResult.load(FPATH)

ptc_curves = PTCtable.ptc_table.groupby(["det_id", "output"])

ids = []
curves = []

for idd, curve in iter(ptc_curves):
    ids.append(idd)
    curves.append(curve)


plt.close("all")


    
