import os

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


script_name = 'surface_deformation'

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [2, 1]
number_of_pixels = (256, 256)


step=0
load='tension' #  'tension'   'compression'
if load=='compression':
    max_load=-0.4
elif load=='tension':
    max_load=0.4
for step in np.arange(0, 50) :

    file_data_name = f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{step}_{load}_max_{max_load}_'

    if step == 0 :

        x_deformed = np.load(data_folder_path + file_data_name + 'X' + f'.npy', allow_pickle=True)
        y_deformed = np.load(data_folder_path + file_data_name + 'Y' + f'.npy', allow_pickle=True)
        file_data_name = f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{1}_{load}_max_{max_load}_'

        phase_field = np.load(data_folder_path + file_data_name + 'phase_field' + f'.npy', allow_pickle=True)
        macro_gradient= np.load(data_folder_path +file_data_name+'macro_gradient'+f'.npy', allow_pickle=True)

    else:
        x_deformed= np.load(data_folder_path +file_data_name+'x_deformed'+f'.npy', allow_pickle=True)

        y_deformed= np.load(data_folder_path +file_data_name+'y_deformed'+f'.npy', allow_pickle=True)

        phase_field = np.load(data_folder_path + file_data_name + 'phase_field' + f'.npy', allow_pickle=True)

    fig = plt.figure(figsize=(6, 3.0))
    gs = fig.add_gridspec(1, 1, hspace=0., wspace=0., width_ratios=[1],
                          height_ratios=[1])

    ax_deformed = fig.add_subplot(gs[0, 0])
    pcm = ax_deformed.pcolormesh(x_deformed, y_deformed, phase_field, cmap=mpl.cm.Greys,vmin=0, vmax=2,
                                 rasterized=True)
    if load == 'compression':
        ax_deformed.set_ylim(0.2,1.15)
        ax_deformed.set_xlim(-0.0, 2.)
    elif load == 'tension':
        ax_deformed.set_ylim(0.2,1.)
        ax_deformed.set_xlim(-0.4, 2.4)
    ax_deformed.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Hide all four spines individually
    ax_deformed.spines['top'].set_visible(False)
    ax_deformed.spines['right'].set_visible(False)
    ax_deformed.spines['bottom'].set_visible(False)
    ax_deformed.spines['left'].set_visible(False)

    ax_deformed.get_xaxis().set_visible(False)  # hides x-axis only
    ax_deformed.get_yaxis().set_visible(False)  # hides y-axis only

    file_data_name = f'{load}_N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{step}_{load}_max_{max_load}_'

    plt.savefig(figure_folder_path+file_data_name+f'.png', bbox_inches='tight',dpi=800)

    plt.show()
    step+=1



#
plot_movie=True
if plot_movie:
    fig = plt.figure(figsize=(6, 3.0))
    #gs = fig.add_gridspec(1, 1)

    ax_deformed = fig.gca() #add_subplot(gs[0, 0])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax_deformed.set_axis_off()


    # Animation function to update the image
    def update(step):
            file_data_name = f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{step}_{load}_max_{max_load}_'

            if step == 0:

                x_deformed = np.load(data_folder_path + file_data_name + 'X' + f'.npy', allow_pickle=True)
                y_deformed = np.load(data_folder_path + file_data_name + 'Y' + f'.npy', allow_pickle=True)
                file_data_name = f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{1}_{load}_max_{max_load}_'

                phase_field = np.load(data_folder_path + file_data_name + 'phase_field' + f'.npy', allow_pickle=True)


            else:
                x_deformed = np.load(data_folder_path + file_data_name + 'x_deformed' + f'.npy', allow_pickle=True)

                y_deformed = np.load(data_folder_path + file_data_name + 'y_deformed' + f'.npy', allow_pickle=True)

                phase_field = np.load(data_folder_path + file_data_name + 'phase_field' + f'.npy', allow_pickle=True)

            ax_deformed.clear()

            ax_deformed.pcolormesh(x_deformed, y_deformed, phase_field, cmap=mpl.cm.Greys,vmin=0, vmax=2,
                                         rasterized=True)
            if load == 'compression':
                ax_deformed.set_ylim(0.2, 1.15)
                ax_deformed.set_xlim(-0.0, 2.)
            elif load == 'tension':
                ax_deformed.set_ylim(0.2, 1.)
                ax_deformed.set_xlim(-0.4, 2.4)
            ax_deformed.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Hide all four spines individually
            ax_deformed.spines['top'].set_visible(False)
            ax_deformed.spines['right'].set_visible(False)
            ax_deformed.spines['bottom'].set_visible(False)
            ax_deformed.spines['left'].set_visible(False)

            ax_deformed.get_xaxis().set_visible(False)  # hides x-axis only
            ax_deformed.get_yaxis().set_visible(False)  # hides y-axis only
            plt.tight_layout()

            #img.set_array(xopt_it)
    # Create animation

    ani = FuncAnimation(fig, update, frames=50, blit=False)
    # Save as a GIF
  #  WriterClass = mpl.animation.writers['ffmpeg']
   # writer = WriterClass(fps=10, metadata=dict(artist='bww'), bitrate=1800)
    video_name = f'{load}_N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_max_{max_load}_'
    ani.save(figure_folder_path+video_name+f'.gif', writer=PillowWriter(fps=10),dpi=1200)
    ani.save(figure_folder_path+video_name+f'.mp4', writer=FFMpegWriter(fps=10),dpi=1200)
