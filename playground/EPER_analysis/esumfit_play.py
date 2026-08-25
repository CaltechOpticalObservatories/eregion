from eregion.core.expsum_fit_math import ExpSumFitter
import numpy as np
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.DEBUG)

#simple test data
xs = np.linspace(0, 22.0, 500)
dU = np.diff(xs)[0]


a_in = [12, 8.0, 3.0]
k_in = [0.3, 0.02, 1.32]

yy = np.zeros_like(xs)
yy += np.random.normal(loc=0.0, scale=0.01, size=len(yy))
                       


for a, k in zip(a_in, k_in):
    yy += a * np.exp(-k * xs)





efitter = ExpSumFitter(yy, dU = dU)

gen = efitter.iterate_fit(None)

i = 0
for thet, a in gen:
#    print(f"thetas: {thet}")
#    print(f"as: {a}")

    ki = -1.0* np.log(thet) / dU

#    print(f"ks: {ki}")


print(f"thetas: {thet}")
print(f"ks: {ki}")
print(f"as: {a}")

xf = np.linspace(0, 1, 200)
yrespoly =  [efitter.residpoly(_) for _ in xf]


yy2 = np.zeros_like(xs)
for k,a in zip(ki, a):
    yy2 += a * np.exp(-k * xs)



plt.close("all")
plt.plot(xs, yy, "--", label="input")
plt.plot(xs, yy2, ".", label="fit")

plt.figure()
plt.plot(xf, yrespoly, ".")
