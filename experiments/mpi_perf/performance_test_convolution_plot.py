import numpy as np
import scipy as sp
import matplotlib as mpl
import time
import sys
import tracemalloc

tracemalloc.start()

sys.path.append("/")
sys.path.append('../..')  # Add parent directory to path

import matplotlib.pyplot as plt

from mpi4py import MPI

plt.rcParams['text.usetex'] = True

if __name__ == '__main__':
    import os
    import time

    file_folder_path = os.path.dirname(os.path.realpath(__file__))

    Ns = [  64, 128, 256, 512, 1024, 2048]  # , 64, 128
    nb_cores =[1,2,4,6,8,10,12]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    random = False
    for i in np.arange(len(nb_cores)):
        core = nb_cores[i]
        elapsed_times = []
        for j in np.arange(len(Ns)):
            N = Ns[j]
            script_name = 'performance_test_convolution' + f'_ranks_{core}' + f'_N_{N}'
            data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'

            _info = np.load(data_folder_path + f'time.npz', allow_pickle=True)

            elapsed_times.append(_info.f.time)
            if i==0:
                single_core=elapsed_times
        ax.semilogx(np.array(Ns) ** 2, np.asarray(elapsed_times)/np.asarray(single_core),  # "k", marker='o',
                  label=f'nb cores = {core}',
                  linewidth=2)
    # ax.loglog(np.array(Ns) ** 2, np.array(Ns) ** 2 / 1e5,  # "k", marker='o',
    #           label=f'linear  ', linestyle='--',
    #           linewidth=2)
    ax.set_xlabel(r"Grid size")
    ax.set_xticks(np.array(Ns) ** 2)
    ax.set_xticklabels([f"${n}^2$" for n in Ns])
    ax.grid(True, which='major', linestyle='-', linewidth=1.2)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
    ax.set_ylabel(r"Time for 10 runs")
    ax.set_title('Time for convolution: apply + transpose ')
    ax.legend()
    fname = file_folder_path + '/figures/' + 'scaling_local{}'.format('.png')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
    if MPI.COMM_WORLD.rank == 0:
        np.savez(data_folder_path + f'time.npz', **_info)

    # plt.imshow(phase_field_0.s[0, 0])
    # plt.show()

    Ns = [256, 512, 1024, 2048, 4096, 8192]  # , 64, 128
    nb_cores = [16, 32, 64, 128]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    random = False
    for i in np.arange(len(nb_cores)):
        core = nb_cores[i]
        elapsed_times = []
        for j in np.arange(len(Ns)):
            N = Ns[j]
            script_name = 'performance_test_convolution' + f'_ranks_{core}' + f'_N_{N}'
            data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'

            _info = np.load(data_folder_path + f'time.npz', allow_pickle=True)

            elapsed_times.append(_info.f.time)
            if i == 0:
                single_core = elapsed_times
        ax.semilogx(np.array(Ns) ** 2, np.asarray(elapsed_times) / np.asarray(single_core),
                    label=f'nb cores = {core}',
                    linewidth=2)
        # ax.loglog(np.array(Ns) ** 2, np.asarray(elapsed_times),  # "k", marker='o',
        #           label=f'nb cores = {core}',
        #           linewidth=2)

    #
    # ax.loglog(np.array(Ns) ** 2, np.array(Ns) ** 2/1e6,  # "k", marker='o',
    #           label=f'linear  ', linestyle='--',
    #           linewidth=2)

    ax.set_xlabel(r"Grid size")
    ax.set_xticks(np.array(Ns) ** 2)
    ax.set_xticklabels([f"${n}^2$" for n in Ns])
    ax.grid(True, which='major', linestyle='-', linewidth=1.2)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
    ax.set_ylabel(r"Time for 10 runs")
    ax.set_title('Time for convolution: apply + transpose ')
    ax.legend()
    fname = file_folder_path + '/figures/' + 'scaling_nemo2{}'.format('.png')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
    if MPI.COMM_WORLD.rank == 0:
        np.savez(data_folder_path + f'time.npz', **_info)

    # plt.imshow(phase_field_0.s[0, 0])
    # plt.show()
