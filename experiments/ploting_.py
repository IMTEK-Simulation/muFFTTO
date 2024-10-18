import netCDF4
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

lstep0 = 'exp_data/last_stepf0.npy'
phase_lstep0 = np.load(lstep0)  # .reshape(8,4)
lstep1 = 'exp_data/last_stepf1.npy'
phase_lstep1 = np.load(lstep1)  # .reshape(8,4)
# fp='exp_data/phase_optimized.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N32_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI4.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N256_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI10.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N256_E_target_0.3_Poisson_0.25_w0.01_eta0.01_p2_bounds=False_FE_NuMPI12.npy'
name = 'muFFTTO_elasticity_random_init_N128_E_target_0.35_Poisson_-0.3_Poisson0_0.1_w0.01_eta0.01_p2_bounds=False_FE_NuMPI1.npy'
for w in  [.02, ]:  #np.arange(0.01, 0.5, 0.05):  # np.arange(0.1, 2.1, 0.1):# for w in np.arange(0.1, 2.1, 0.1):

    for eta_mult in [.1, ]:  #np.arange(0.01, 0.5, 0.05):#[0.02, ]:# np.arange(1, 2, 1):  # for eta_mult in np.arange(1, 5, 1):

        print(w, eta_mult)
        pixel_size = 0.0078125
        eta = 0.03125#eta_mult * pixel_size
        N=8
        cores=2
        p=2
        name = (
            f'1muFFTTO_elasticity_random_init_N{N}_E_target_0.25_Poisson_-0.5_Poisson0_0.2_w{w}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}.npy')

        #  '1muFFTTO_elasticity_random_init_N256_E_target_0.35_Poisson_-0.3_Poisson0_0.0_w0.02_eta0.00390625_p2_bounds=False_FE_NuMPI12.npy'
        # "1muFFTTO_elasticity_random_init_N258_E_target_0.35_Poisson_-0.3_Poisson0_0.0_w1.0_eta0.003875968992248062_p2_bounds=False_FE_NuMPI12.npy"
        fp = 'exp_data/' + name
        #   '1muFFTTO_elasticity_random_init_N128_E_target_0.35_Poisson_-0.3_Poisson0_0.0_w0.4_eta0.03125_p2_bounds=False_FE_NuMPI10.npy'

        # fp='exp_data/muFFTTO_elasticity_random_init_N64_E_target_0.3_Poisson_0.25_w0.01_eta0.01_p2_bounds=False_FE_NuMPI6.npy'
        #

        phase_field = np.load(fp)
        # sensitivity = np.load(f'sensitivity'+ fp)

        log_name = (
            f'1muFFTTO_elasticity_random_init_N{N}_E_target_0.25_Poisson_-0.5_Poisson0_0.2_w{w}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}.npyxopt_log.npz')
        #    '1muFFTTO_elasticity_random_init_N256_E_target_0.35_Poisson_-0.3_Poisson0_0.0_w0.01_eta0.00390625_p2_bounds=False_FE_NuMPI12.npyxopt_log.npz'

        fp_log = 'exp_data/' + log_name
        # xopt=np.load('exp_data/' + log_name)
        xopt = np.load(fp_log)

        src = './figures/'  # source folder\
        fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        plt.figure()
        plt.contourf(np.tile(phase_field, (3, 3)), cmap=mpl.cm.Greys)
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f'w = {w},eta= {eta_mult}\n, {xopt.f.message}')
        fname = src + fig_data_name + '{}'.format('.png')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, bbox_inches='tight')
        plt.show()

quit()

plt.figure()
fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

plt.plot(np.tile(phase_field, (1, 2))[40, :].transpose())
# nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
plt.grid(True)
plt.minorticks_on()
fname = src + fig_data_name + '{}'.format('.png')
print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
plt.savefig(fname, bbox_inches='tight')

plt.show()
quit()
plt.figure()
# plt.contourf(phase_field_sol_FE_MPI[0, 0], cmap=mpl.cm.Greys)
# nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
# plt.clim(0, 1)
plt.colorbar()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# N=512
rho = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 2, 10, 100, 1000, 10000, 100000])
# x =

error_16 = np.array(
    [0.008423935620133993, 0.00841991604982395, 0.008379851394533566, 0.007991942675222075,
     0.0050983065833956065, 0.0010791753165280138, 0.012157500314251113, 0.023355094613997762,
     0.025072715887334285, 0.025253014236804816, 0.025271132965769327])
error_32 = np.array(
    [0.003366687704723814, 0.0033648041231111314, 0.0033460407590564234, 0.0031654214234443367,
     0.0018836197000315913, .0003426444017404773, 0.004491708515460546, 0.009250406295694402,
     0.010011434015880338, 0.010091721333346015, 0.010099793787232914])

error_128 = np.array(
    [0.0005319756234268835, 0.0005315840068030875, 0.0005276873020660933, 0.0004905900993198431,
     0.00025083171168960305, 3.199681503196494e-05, 0.0005981371586438744, 0.0014336662125764565,
     0.0015788530343969764, 0.001594326880746122, 0.001595884313507101])

error_512 = np.array([8.38180125228849e-05, 8.374106730946185e-05, 8.297626632713939e-05, 7.577188452034811e-05,
                      3.3079980324535185e-05, 2.8367217486113816e-06, 7.888303000469499e-05, 0.0002511562291653835,
                      0.0002482669743246735, 0.0002214304586469762, 0.00025144733232673744])

error_1024 = np.array([
    3.32645214683458e-05, 3.3230943471851404e-05, 3.28973701249069e-05, 2.977135369508499e-05,
    1.2001433427166752e-05, 8.358510967809707e-07, 2.8618802789148745e-05, 8.700172293418795e-05,
    9.842971858975424e-05, 9.966625363388992e-05, 9.979090332401519e-05])

plt.loglog(rho, error_16, '|-', label='N=16')

plt.loglog(rho, error_32, '|-', label='N=32')
plt.loglog(rho, error_128, 'o-', label='N=128')
plt.loglog(rho, error_512, 'x--', label='N=512')
plt.loglog(rho, error_1024, '>--', label='N=1024')

plt.xlabel(r'phase contrast $\rho$')
plt.ylabel(r'Total error in  homogenized data $A_{11}^{FEM}-A_{11}^{Analytical}$')
plt.legend(loc='best')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# N=512
rho = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 2, 10, 100, 1000, 10000, 100000])
# x =


error_32 = np.array(
    [0.003366687704723814, 0.0033648041231111314, 0.0033460407590564234, 0.0031654214234443367,
     0.0018836197000315913, .0003426444017404773, 0.004491708515460546, 0.009250406295694402,
     0.010011434015880338, 0.010091721333346015, 0.010099793787232914])

error_512 = np.array([8.38180125228849e-05, 8.374106730946185e-05, 8.297626632713939e-05, 7.577188452034811e-05,
                      3.3079980324535185e-05, 2.8367217486113816e-06, 7.888303000469499e-05, 0.0002511562291653835,
                      0.0002482669743246735, 0.0002214304586469762, 0.00025144733232673744])

plt.loglog(rho, error_32, ':', label='N=32 top', marker='|', markersize=15)
plt.loglog(rho, error_32, ':', label='N=32 bottom', marker='o', markerfacecolor='none')
plt.loglog(rho, error_512, '--', label='N=512 top', marker='|', markersize=15)
plt.loglog(rho, error_512, '--', label='N=512 bottom', marker='o', markerfacecolor='none')

plt.xlabel(r'phase contrast $\rho$')
plt.ylabel(r'Total error in  homogenized data $A_{11}^{FEM}-A_{11}^{Analytical}$')
plt.legend(loc='best')
plt.show()
