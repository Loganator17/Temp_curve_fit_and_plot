import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

fname = "/InSAR/Bardarbunga/time_series_analysis/path09/mintpy_ids/FUNCTION_fitter/data_formated_mm.txt"
cero = np.genfromtxt(fname, names="orbit_day, point1, point2, point3, point4, point5", dtype=None)
x = np.array(cero["orbit_day"])
y2 = np.array(cero["point1"])
y3 = np.array(cero["point2"])
y4 = np.array(cero["point3"])
y5 = np.array(cero["point4"])
y6 = np.array(cero["point5"])


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
# perform the fit
p0 = (0, .001, 800) # start with values near those we expect
params1, cv1 = scipy.optimize.curve_fit(monoExp, x, y2, p0)
params2, cv2 = scipy.optimize.curve_fit(monoExp, x, y3, p0)
params3, cv3 = scipy.optimize.curve_fit(monoExp, x, y4, p0)
params4, cv4 = scipy.optimize.curve_fit(monoExp, x, y5, p0)
params5, cv5 = scipy.optimize.curve_fit(monoExp, x, y6, p0)
m1, t1, b1 = params1
m2, t2, b2 = params2
m3, t3, b3 = params3
m4, t4, b4 = params4
m5, t5, b5 = params5
sampleRate = 20_000 # Hz
tauSec1 = (1 / t1) / sampleRate
tauSec2 = (1 / t2) / sampleRate
tauSec3 = (1 / t3) / sampleRate
tauSec4 = (1 / t4) / sampleRate
tauSec5 = (1 / t5) / sampleRate

# determine quality of the fit
squaredDiffs1 = np.square(y2 - monoExp(x, m1, t1, b1))
squaredDiffs2 = np.square(y3 - monoExp(x, m2, t2, b2))
squaredDiffs3 = np.square(y4 - monoExp(x, m3, t3, b3))
squaredDiffs4 = np.square(y5 - monoExp(x, m4, t4, b4))
squaredDiffs5 = np.square(y6 - monoExp(x, m5, t5, b5))

squaredDiffsFromMean1 = np.square(y2 - np.mean(y2))
squaredDiffsFromMean2 = np.square(y3 - np.mean(y3))
squaredDiffsFromMean3 = np.square(y4 - np.mean(y4))
squaredDiffsFromMean4 = np.square(y5 - np.mean(y5))
squaredDiffsFromMean5 = np.square(y6 - np.mean(y6))
rSquared1 = 1 - np.sum(squaredDiffs1) / np.sum(squaredDiffsFromMean1)
rSquared2 = 1 - np.sum(squaredDiffs2) / np.sum(squaredDiffsFromMean2)
rSquared3 = 1 - np.sum(squaredDiffs3) / np.sum(squaredDiffsFromMean3)
rSquared4 = 1 - np.sum(squaredDiffs4) / np.sum(squaredDiffsFromMean4)
rSquared5 = 1 - np.sum(squaredDiffs5) / np.sum(squaredDiffsFromMean5)
print(f"R² = {rSquared1}")
print(f"R² = {rSquared2}")
print(f"R² = {rSquared3}")
print(f"R² = {rSquared4}")
print(f"R² = {rSquared5}")


x2=np.linspace(0, np.max(x), 1000)
# plot the results
plt.plot(x, y2, '.', c='blue')
plt.plot(x2, monoExp(x2, m1, t1, b1), c='blue')
plt.plot(x, y3, '.', c='magenta')
plt.plot(x2, monoExp(x2, m2, t2, b2), c='magenta')
plt.plot(x, y4, '.', c='red')
plt.plot(x2, monoExp(x2, m3, t3, b3), c='red')
plt.plot(x, y5, '.', c='purple')
plt.plot(x2, monoExp(x2, m4, t4, b4), c='purple')
plt.plot(x, y6, '.', c='cornflowerblue')
plt.plot(x2, monoExp(x2, m5, t5, b5), c='cornflowerblue')

plt.plot(x, y2, '.', c='blue', label='64.906, -16.718 (1)')
plt.plot(x, y3, '.', c='magenta', label='64.903, -16.636 (2)')
plt.plot(x, y4, '.', c='red', label='64.889, -16.711 (3)')
plt.plot(x, y5, '.', c='purple', label='64.882, -16.663 (4)')
plt.plot(x, y6, '.', c='cornflowerblue', label='64.918, -16.745 (5)')


#Adjust the various labels in the plot so they do not overlap 
plt.title('Time-Series of Holuhraun 2015-2020 path 147 \n Raw Displacements') #, GIA+PV adjusted')
#plt.ylabel("Change in Kelvin ($^\circ$K)")
plt.legend()
plt.ylabel("Displacement in LOS (mm)")
plt.show()

# inspect the parameters
print(f"Y2 = {m1} * e^(-{t1} * x) + {b1}")
print(f"Y3 = {m2} * e^(-{t2} * x) + {b2}")
print(f"Y4 = {m3} * e^(-{t3} * x) + {b3}")
print(f"Y5 = {m4} * e^(-{t4} * x) + {b4}")
print(f"Y6 = {m5} * e^(-{t5} * x) + {b5}")



gamma=1.7
alpha=.0205736431 #8.3*np.exp(-6)
height1 = 17.9538
height2 = 25.8428
height3 = 26.4292
height4 = 13.9528
height5 = 3.83122

a = 152.8446238947033
k = -0.0013138923323473964
b = -181.1161308255617

dT1 = []
for Px in x2:
     dh = m1*np.exp(Px*-t1) + b1
     dT = np.array(dh / (gamma * alpha * height1))
     dT1.append(dT)

dT2 = []
for Px in x2:
     dh = m2*np.exp(Px*-t2) + b2
     dT = np.array(dh / (gamma * alpha * height2))
     dT2.append(dT)

dT3 = []
for Px in x2:
     dh = m3*np.exp(Px*-t3) + b3
     dT = np.array(dh / (gamma * alpha * height3))
     dT3.append(dT)

dT4 = []
for Px in x2:
     dh = m4*np.exp(Px*-t4) + b4
     dT = np.array(dh / (gamma * alpha * height4))
     dT4.append(dT)

dT5 = []
for Px in x2:
     dh = m5*np.exp(Px*-t5) + b5
     dT = np.array(dh / (gamma * alpha * height5))
     dT5.append(dT)

plt.plot(x2, dT1, c='blue', label='64.906, -16.718 (1)')
plt.plot(x2, dT2, c='magenta', label='64.903, -16.636 (2)')
plt.plot(x2, dT3, c='red', label='64.889, -16.711 (3)')
plt.plot(x2, dT4, c='purple', label='64.882, -16.663 (4)')
plt.plot(x2, dT5, c='cornflowerblue', label='64.918, -16.745 (5)')

plt.title('Exponential Temperature Change 2015-2020 path 147 \n Raw Displacements') #, GIA+PV adjusted')
plt.ylabel("Change in Kelvin ($^\circ$K)")
plt.xlabel("Time Series Days")
plt.legend()
plt.tight_layout()

plt.show()

#EOF
