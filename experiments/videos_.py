import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpmath import arange
from matplotlib.animation import FuncAnimation, PillowWriter

lstep0 = 'exp_data/last_stepf0.npy'
phase_lstep0 = np.load(lstep0)  # .reshape(8,4)
lstep1 = 'exp_data/last_stepf1.npy'
phase_lstep1 = np.load(lstep1)  # .reshape(8,4)
# fp='exp_data/phase_optimized.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N32_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI4.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N256_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI10.npy'
# fp='exp_data/muFFTTO_elasticity_random_init_N256_E_target_0.3_Poisson_0.25_w0.01_eta0.01_p2_bounds=False_FE_NuMPI12.npy'

for w_mult in [
    4.0]:  # np.arange(0.1, 1., 0.1):# [1]:  # np.arange(1, 5,1):#3[1]:  # np.arange(1, 5,1):#[1]:  #np.arange(0.01, 0.5, 0.05):  # np.arange(0.1, 2.1, 0.1):# for w in np.arange(0.1, 2.1, 0.1):

    for eta_mult in [
        0.01, ]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:  #np.arange(0.001, 0.01, 0.002):#[0.005, ]:  #np.arange(0.01, 0.5, 0.05):#[0.02, ]:# np.arange(1, 2, 1):  # for eta_mult in np.arange(1, 5, 1):
        energy_objective = False
        print(w_mult, eta_mult)
        pixel_size = 0.0078125
        eta = 0.03125  # eta_mult * pixel_size
        N = 64
        cores = 6
        p = 2
        nb_load_cases = 3
        random_initial_geometry = True
        bounds=False




# Generate example data (list of 2D numpy arrays)
#frames = [np.random.rand(10, 10) for _ in range(20)]

# Set up the figure
#fig, ax = plt.subplots()
#img = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
# log_name_it = (
#         f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{1}.npy')
# log_name_it = (
#         f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.7_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{1}.npy')
log_name_it = (
        f'adam_muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{1}.npy')

xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
#cont = plt.contourf(xopt_it,  cmap='viridis', levels=20)


fig = plt.figure()
ax = plt.axes(xlim=(0, N), ylim=(0, N))
# Animation function to update the image
def update(i):
    #img.set_array(frames[i])
    #global cont
    # log_name_it = (        f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    #
    # log_name_it = (    #     f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.7_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    # log_name_it = (
    #     f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    log_name_it = (
        f'adam_muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')

    xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)

    ax.clear()
    ax.imshow(xopt_it,  cmap=mpl.cm.Greys, vmin=0, vmax=1)

    ax.set_title('iteration {}'.format(i))
    #img.set_array(xopt_it)
# Create animation
ani = FuncAnimation(fig, update, frames=686, blit=False)

# Save as a GIF
ani.save(f"./figures/movieN{N}3_exp2_imshow_adam.gif", writer=PillowWriter(fps=40))


fig = plt.figure()
ax = plt.axes(xlim=(0, 3*N), ylim=(0, 3*N))
# Animation function to update the image
def update(i):
    log_name_it = (
        f'adam_muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')

    xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
    ax.clear()
    ax.imshow(np.tile(xopt_it, (3, 3)),  cmap=mpl.cm.Greys, vmin=0, vmax=1)
    #ax.clim(0, 1)
    #ax.colorbar()
    ax.set_title('iteration {}'.format(i))
    #img.set_array(xopt_it)
# Create animation
ani = FuncAnimation(fig, update, frames=686, blit=False)
# Save as a GIF
ani.save(f"./figures/movie3x3N{N}3_exp2_imshow_adam.gif", writer=PillowWriter(fps=40))


fig = plt.figure()
ax = plt.axes(xlim=(0, N), ylim=(0, N))
# Animation function to update the image
def update(i):
    #img.set_array(frames[i])
    #global cont
    # log_name_it = (        f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    #
    # log_name_it = (    #     f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.7_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    # log_name_it = (
    #     f'eta_3muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
    log_name_it = (
        f'lbfg_muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')

    xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
    ax.clear()
    ax.imshow(xopt_it,  cmap=mpl.cm.Greys, vmin=0, vmax=1)

    ax.set_title('iteration {}'.format(i))
# Create animation
ani = FuncAnimation(fig, update, frames=471, blit=False)
# Save as a GIF
ani.save(f"./figures/movieN{N}3_exp2_imshow_lbfg.gif", writer=PillowWriter(fps=40))


fig = plt.figure()
ax = plt.axes(xlim=(0, 3*N), ylim=(0, 3*N))
# Animation function to update the image
def update(i):
    log_name_it = (
        f'lbfg_muFFTTO_elasticity_random_init_N{N}_E_target_0.3_Poisson_0.0_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')

    xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
    ax.clear()
    ax.imshow(np.tile(xopt_it, (3, 3)),  cmap=mpl.cm.Greys, vmin=0, vmax=1)
    #ax.clim(0, 1)
    #ax.colorbar()
    ax.set_title('iteration {}'.format(i))

# Create animation
ani = FuncAnimation(fig, update, frames=395, blit=False)

# Save as a GIF
ani.save(f"./figures/movie3x3N{N}3_exp2_imshow_lbfg.gif", writer=PillowWriter(fps=40))






quit()
# initializing a figure in
# which the graph will be plotted
fig = plt.figure()

# marking the x-axis and y-axis
axis = plt.axes(xlim=(0, 4),
                ylim=(-2, 2))

# initializing a line variable
line, = axis.plot([], [], lw=3)


# data which the line will
# contain (x, y)
def init():
    line.set_data([], [])

    return line,




def animate(i):
    x = np.linspace(0, 4, 1000)

    # plots a sine graph
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)




    return line,


anim = FuncAnimation(fig, animate,
                     init_func=init,
                     frames=200,
                     interval=20,
                     blit=True)

anim.save('./figures/continuousSineWave.gif',
          writer='pillow', fps=30)

# which the graph will be plotted
fig, ax= plt.subplots()

# initializing a line variable
#line, = axis.plot([], [], lw=3)
log_name_it = (
        f'eta_2muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{1}.npy')
xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
img = ax.contourf(xopt_it)


# data which the line will
# contain (x, y)
def init():
    #line.set_data([], [])
    img = ax.contourf(xopt_it)
    return img


def animate(i):
     log_name_it = (
        f'eta_2muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i + 1}.npy')
     xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
     img =ax.set_array(xopt_it)
     return img


anim = FuncAnimation(fig, animate,
                     init_func=init,
                     frames=200,
                     interval=20,
                     blit=True)

anim.save('./figures/dddd.gif',
          writer='pillow', fps=30)

quit()











# for iteration in np.arange(1, 714, 5, dtype=int):
#     log_name_it = (
#         f'eta_2muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{iteration}.npy')
#     #    '1muFFTTO_elasticity_random_init_N256_E_target_0.35_Poisson_-0.3_Poisson0_0.0_w0.01_eta0.00390625_p2_bounds=False_FE_NuMPI12.npyxopt_log.npz'
#
#     xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
#     plt.contourf(xopt_it, cmap=mpl.cm.Greys)
#     # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
#     plt.clim(0, 1)
#     plt.colorbar()
#
#     plt.show()

        # Set up the figure
fig, ax = plt.subplots()


log_name_it = (
        f'eta_2muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{1}.npy')
xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)




#ax.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
img = ax.contourf(xopt_it,  cmap=mpl.cm.Greys)
           # .contourf(xopt_it, cmap=mpl.cm.Greys)

# Animation function to update the image
def update(i):
    log_name_it = (
        f'eta_2muFFTTO_elasticity_random_init_N{N}_E_target_0.33333333333333337_Poisson_-0.3333333333333333_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{i+1}.npy')
    xopt_it = np.load('exp_data/' + log_name_it, allow_pickle=True)
    img.set_array(xopt_it)
    return [img]


# Create animation
ani = FuncAnimation(fig, update, frames=20, interval=1, blit=True)

# Save as a GIF

#ani.save("movie.gif", writer=PillowWriter(fps=5))

ani.save(filename="./figures/pillow_example.mp4",  writer = 'pillow', fps = 5)

quit()

plt.figure()
fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

plt.plot(np.tile(phase_field, (1, 2))[:, 50].transpose())
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
