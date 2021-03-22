
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import misc
from scipy.special import cbrt
from scipy.special import exp10

import numpy as np
import pandas as pd

array = np.array([0, 1, 2, 3])
print(array)

cb = cbrt([27, 64])
print(cb)

exp = exp10([1, 10])
print(exp)

two_d_array = np.array([ [4, 5], [3, 2] ])
linalg.det(two_d_array)

two_d_array = np.array([ [4, 5], [3, 2] ])
linalg.inv(two_d_array)

# define two dimensional array
arr = np.array([[5, 4], [6, 3]])
# pass value into function
eg_val, eg_vect = linalg.eig(arr)
# get eigenvalues
print(eg_val)
# get eigenvectors
print(eg_vect)

# Frequency in terms of Hertz
fre = 5 
# Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint=False)
a = np.sin(fre * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
plt.show()

# get face image of panda from misc package
panda = misc.face()
# plot or show image of face
plt.imshow(panda)
plt.show()

# pandas (Use Anaconda-Spyder)

