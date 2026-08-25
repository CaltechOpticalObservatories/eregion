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
yy += np.random.normal(loc=0.0, scale=0.04, size=len(yy))
                       


for a, k in zip(a_in, k_in):
    yy += a * np.exp(-k * xs)





efitter = ExpSumFitter(yy, dU = dU)


efitter.run_fit()

thet = efitter.thetas
a = efitter.a 
ki = -1.0* np.log(thet) / dU


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



for reft, refa in efitter.simple_coalesce():
    print(f"reft: {reft}, refa: {refa}")
    refk = -np.log(reft) / dU
    print(f"refk: {refk}")

yy3 = np.zeros_like(xs)
for k,a in zip(refk, refa):
    yy3 += a * np.exp(-k * xs)

plt.plot(xs, yy3, "x", label="coalesced terms only")
    

plt.figure()
plt.plot(xf, yrespoly, ".")

