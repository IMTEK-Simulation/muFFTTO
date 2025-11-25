import numpy as np
import scipy as sc
import time

import matplotlib as mpl
from matplotlib import pyplot as plt






N_x=2**np.array([5,6,7,8,9])
time_GJ= np.array([3.950589179992676,13.753841161727905,69.46913194656372,402.0402536392212,2189.5818433761597])
time_G= np.array([4.966899633407593, 31.238507747650146,163.86940717697144,1112.1157495975494,5965.060667276382])
nb_iter_GJ=np.array([219,286,375,510,690])
nb_iter_G=np.array([450,743,1063,1466,1969])


fig = plt.figure(figsize=(9, 3.0))
gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                      height_ratios=[1])

plt.loglog(3*N_x**2, time_GJ/60, 'b-', label='GJ')
plt.loglog(3*N_x**2, time_G/60, 'r-', label='G')
plt.loglog(3*N_x**2,  N_x**2/1e4, '--', label='linear')
plt.loglog(3*N_x**2,  N_x**2 * np.log(N_x**2)/1e5, ':', label='N log N')
plt.xlabel('N Dofs')
plt.ylabel('Time (s)')
plt.legend(loc='best')
plt.show()


fig = plt.figure(figsize=(9, 3.0))
gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                      height_ratios=[1])

plt.loglog(3*N_x**2, nb_iter_GJ, 'b-', label='GJ')
plt.loglog(3*N_x**2, nb_iter_G , 'r-', label='G')
plt.xlabel('N Dofs')
plt.ylabel('nb_iter_G')
plt.legend(loc='best')
plt.show()

fig = plt.figure(figsize=(9, 3.0))
gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                      height_ratios=[1])

plt.loglog(3*N_x**2, time_GJ/nb_iter_GJ, 'b-', label='GJ')
plt.loglog(3*N_x**2, time_G/nb_iter_G, 'r-', label='G')
plt.loglog(3*N_x**2,  N_x**2/1e5, '--', label='linear')
plt.loglog(3*N_x**2,  N_x**2 * np.log(N_x**2)/1e6, ':', label='N log N')
plt.xlabel('N Dofs')
plt.ylabel('Time (s)/ nb CG iterations')
plt.legend(loc='best')
plt.show()


#
'''
macro_gradient_inc[0, 1] += 0.05 / float(ninc)
macro_gradient_inc[1, 0] += 0.05 / float(ninc)

element_type :  trilinear_hexahedron
number_of_pixels:  (32, 32, 1)
preconditioner_type: Green
Total number of CG 450
Total number of sum_Newton_its 7
Elapsed time :  4.966899633407593
Elapsed time:  0.08278166055679322

element_type :  trilinear_hexahedron
number_of_pixels:  (64, 64, 1)
preconditioner_type: Green
Total number of CG 743
Total number of sum_Newton_its 7
Elapsed time :  31.238507747650146
Elapsed time:  0.5206417957941691

element_type :  trilinear_hexahedron
number_of_pixels:  (128, 128, 1)
preconditioner_type: Green
Total number of CG 1063
Total number of sum_Newton_its 7
Elapsed time :  163.86940717697144
Elapsed time:  2.7311567862828574

element_type :  trilinear_hexahedron
number_of_pixels:  (256, 256, 1)
preconditioner_type: Green
Total number of CG 1466
Total number of sum_Newton_its 7
Elapsed time :  1112.1157495975494
Elapsed time:  18.53526249329249

element_type :  trilinear_hexahedron
number_of_pixels:  (512, 512, 1)
preconditioner_type: Green
Total number of CG 1969
Total number of sum_Newton_its 7
Elapsed time :  5965.060667276382
Elapsed time:  99.41767778793971


element_type :  trilinear_hexahedron
number_of_pixels:  (32, 32, 1)
preconditioner_type: Jacobi_Green
Total number of CG 219
Total number of sum_Newton_its 7
Elapsed time :  3.950589179992676
Elapsed time:  0.06584315299987793


element_type :  trilinear_hexahedron
number_of_pixels:  (64, 64, 1)
preconditioner_type: Jacobi_Green
Total number of CG 286
Total number of sum_Newton_its 7
Elapsed time :  13.753841161727905
Elapsed time:  0.22923068602879842

element_type :  trilinear_hexahedron
number_of_pixels:  (128, 128, 1)
preconditioner_type: Jacobi_Green
Total number of CG 375
Total number of sum_Newton_its 7
Elapsed time :  69.46913194656372
Elapsed time:  1.157818865776062


element_type :  trilinear_hexahedron
number_of_pixels:  (256, 256, 1)
preconditioner_type: Jacobi_Green
Total number of CG 510
Total number of sum_Newton_its 7
Elapsed time :  402.0402536392212
Elapsed time:  6.70067089398702

element_type :  trilinear_hexahedron
number_of_pixels:  (512, 512, 1)
preconditioner_type: Jacobi_Green
Total number of CG 690
Total number of sum_Newton_its 7
Elapsed time :  2189.5818433761597
Elapsed time:  36.49303072293599





'''
