from eregion.core.expsum_fit_math import ExpSumFitter
import numpy as np
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.DEBUG)

#simple test data
xs = np.linspace(0, 9.0, 100)
dU = np.diff(xs)[0]
y1 = 12* np.exp(-0.3 * xs)
#y2 = 3 * np.exp(-0.02 * xs)

#yy = y1 + y2

yy = y1


efitter = ExpSumFitter(yy, dU = dU)

gen = efitter.iterate_fit(None)

i = 0
for thet, a in gen:
    print(f"thetas: {thet}")
    print(f"as: {a}")

    ki = -1.0* np.log(thet) / dU

    print(f"ks: {ki}")
    

xf = np.linspace(0, 1, 200)
yrespoly =  [efitter.residpoly(_) for _ in xf]


yy2 = np.zeros_like(xs)
for k,a in zip(ki, a):
    yy2 += a * np.exp(-k * xs * dU)



