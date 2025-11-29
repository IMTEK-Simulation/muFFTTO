import numpy as np
import scipy as sc
import time
import os
import sys
import gc
import argparse

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')

from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers

script_name = os.path.splitext('exp_paper_JG_2D_elasticity_TO_1024')[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_2D_elasticity_TO_1024_plot.py",
    description="Solve non-linear elasticity example "
                "from J.Zeman et al., Int. J. Numer. Meth. Engng 111, 903–926 (2017)."
)
parser.add_argument("-n", "--nb_pixel", default="64")
parser.add_argument("-it", "--iteration", default="1")

parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green",
    help="Type of preconditioner to use"
)
args = parser.parse_args()

n_pix = int(args.nb_pixel)
number_of_pixels = (n_pix, n_pix)  # (1024, 1024)
iteration=args.iteration
preconditioner_type = args.preconditioner_type

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'


import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["text.usetex"] = True
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"
nb_iterations_G = []
nb_iterations_J = []
nb_iterations_GJ = []
for iteration in np.arange(1, 97):
    info_log_final_G = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_it_{iteration}.npz', allow_pickle=True)
    nb_iterations_G.append(len(info_log_final_G.f.norm_rr))

    info_log_final_J = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Jacobi_it_{iteration}.npz', allow_pickle=True)
    nb_iterations_J.append(len(info_log_final_J.f.norm_rr))

    info_log_final_GJ = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_Jacobi_it_{iteration}.npz', allow_pickle=True)
    nb_iterations_GJ.append(len(info_log_final_GJ.f.norm_rr))

    print()
nb_iterations_G = np.array(nb_iterations_G)
nb_iterations_J = np.array(nb_iterations_J)
nb_iterations_GJ = np.array(nb_iterations_GJ)

# fig = plt.figure(figsize=(11.5, 6))
fig = plt.figure(figsize=(8.3, 6.1))

plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.2]
                      , height_ratios=[1, 1.7], hspace=0.07)
ax_iterations = fig.add_subplot(gs[1:, :])
ax_iterations.text(-0.1, 1.0, rf'\textbf{{(b)}}', transform=ax_iterations.transAxes)

ax_iterations.plot(np.linspace(1, 1000, nb_iterations_G.shape[0]), nb_iterations_G, "g", label='Green N=64', linewidth=1)
ax_iterations.plot(np.linspace(1, 1000, nb_iterations_J.shape[0]), nb_iterations_J, "b", label='Jacobi N=64', linewidth=1)
# ax_iterations.plot(nb_iterations_J, "b", label='Jacobi N=64', linewidth=1)

ax_iterations.plot(np.linspace(1, 1000, nb_iterations_GJ.shape[0]), nb_iterations_GJ, "k", label='Green-Jacobi  N=64', linewidth=2)
#
# ax_iterations.plot(np.linspace(1, 1000, dgo_32.shape[0]), dgo_32, "g", label='Green N=32', linewidth=1,
#                    linestyle=':')
# ax_iterations.plot(np.linspace(1, 1000, jacoby_32.shape[0]), jacoby_32, "b", label='Jacobi N=32', linewidth=1,
#                    linestyle=':')
# ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), combi_32, "k", label='Jacobi - Green N=32', linewidth=2,
# linestyle=':')
plt.show()
