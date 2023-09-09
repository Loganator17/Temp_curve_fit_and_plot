import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

fname = "/InSAR/Bardarbunga/time_series_analysis/path09/mintpy_ids/FUNCTION_fitter/data_formated_mm.txt"
cero = np.genfromtxt(fname, names="orbit_day, point1, point2, point3, point4, point5", dtype=None)
x = np.array(cero["orbit_day"])
y2 = np.array(cero["point1"])

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
# perform the fit
p0 = (0, .001, 800) # start with values near those we expect
params1, cv1 = scipy.optimize.curve_fit(monoExp, x, y2, p0)

m1, t1, b1 = params1

sampleRate = 20_000 # Hz
tauSec1 = (1 / t1) / sampleRate

# determine quality of the fit
squaredDiffs1 = np.square(y2 - monoExp(x, m1, t1, b1))

squaredDiffsFromMean1 = np.square(y2 - np.mean(y2))

rSquared1 = 1 - np.sum(squaredDiffs1) / np.sum(squaredDiffsFromMean1)

print(f"R² = {rSquared1}")



x2=np.linspace(0, np.max(x), 1000)
# plot the results
plt.plot(x, y2, '.', c='blue')
plt.plot(x2, monoExp(x2, m1, t1, b1), c='blue')


plt.plot(x, y2, '.', c='blue', label='64.906, -16.718 (1)')


#Adjust the various labels in the plot so they do not overlap 
plt.title('Time-Series of Holuhraun 2015-2020 path 147 \n Raw Displacements') #, GIA+PV adjusted')
#plt.ylabel("Change in Kelvin ($^\circ$K)")
plt.legend()
plt.ylabel("Displacement in LOS (mm)")
plt.show()

# inspect the parameters
print(f"Y2 = {m1} * e^(-{t1} * x) + {b1}")



gamma=1.7
alpha=.0205736431 #8.3*np.exp(-6)
height1 = 17.9538


a = 152.8446238947033
k = -0.0013138923323473964
b = -181.1161308255617

dT1 = []
for Px in x2:
     dh = m1*np.exp(Px*-t1) + b1
     dT = np.array(dh / (gamma * alpha * height1))
     dT1.append(dT)


plt.plot(x2, dT1, c='blue', label='64.906, -16.718 (1)')

plt.title('Exponential Temperature Change 2015-2020 path 147 \n Raw Displacements') #, GIA+PV adjusted')
plt.ylabel("Change in Kelvin ($^\circ$K)")
plt.xlabel("Time Series Days")
plt.legend()
plt.tight_layout()

plt.show()

#EOF
