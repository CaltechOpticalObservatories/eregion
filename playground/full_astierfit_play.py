import numpy as np
from astropy.io import fits
from astropy.table import Table

import pyqtgraph as pg
import matplotlib.pyplot as plt


hdul = fits.open("PTC_NOCTI_d2.fits")

tab = Table(hdul[1].data)
sel = tab["output"] == "F"


mu = tab[sel]["mean"]
psd = tab[sel]["PSD_0-1"]

covs = np.real(np.fft.ifft2(psd))


correls = covs / covs[:,0,0][:, np.newaxis, np.newaxis]

cutdiffmed = np.std(tab[sel]["mean_0-1"])
sel2 = np.abs(tab[sel]["mean_0-1"]) < 3 * cutdiffmed
