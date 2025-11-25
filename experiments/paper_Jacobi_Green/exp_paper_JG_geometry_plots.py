import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import precision

# plot geometries
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library


def scale_field(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_min, field_max = field.min(), field.max()
    scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
    return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]

src = './figures/'
# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# Define the number of elements in x and y directions
def get_triangle(nx= 5,ny= 5,lx=1.0, ly=1.0 ):


    # Generate grid points
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    # Create triangular connectivity
    triangles = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            triangles.append([n1, n2, n3])  # First triangle
            triangles.append([n2, n4, n3])  # Second triangle

    triangles = np.array(triangles)
    return triangles, X, Y

if __name__ == '__main__':

    triangles, X, Y= get_triangle(nx=5,ny= 5)
    # Create the triangulation object
    triangulation = tri.Triangulation(X, Y, triangles)

    # Plot the mesh
    plt.figure(figsize=(6, 6))
    plt.triplot(triangulation, 'k-', lw=0.8)
    plt.scatter(X, Y, color='red', s=10)  # Plot nodes
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regular FEM Mesh")
    plt.gca().set_aspect('equal')  # Keep aspect ratio
    plt.show()

    #
    nb_pix_multips = [4,3,2 ]
    geometry_ID = 'sine_wave_'  # 'sine_wave_' n_laminate  linear  # 'abs_val' sine_wave_   ,laminate_log  geometry_ID = 'right_cluster_x3'  # laminate2       # 'abs_val' sine_wave_   ,laminate_log

    counter = 0
    ratio=1
    #fig = plt.figure(figsize=(5, 5))
    fig = plt.figure(figsize=(4.15, 4.15))

    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    gs = fig.add_gridspec(3, 3, hspace=0.1, wspace=0.1, width_ratios=[1 , 1, 1 ],
                          height_ratios=[1, 1,1])
    for G_a in np.arange(np.size(nb_pix_multips)):

        for T_a in np.arange(np.size(nb_pix_multips[0:G_a+1])):
            print(f'Ga={G_a}')
            print(f'Ta={T_a}')
            nb_pix_multip = nb_pix_multips[G_a]
           # print(f'nb_pix_multip = {nb_pix_multip}')
            # system set up
            number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)

            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            # if kk == 0:
            if geometry_ID == 'laminate_log' or geometry_ID == 'n_laminate':
                phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords,
                                                                  seed=1,
                                                                  parameter=number_of_pixels[0],
                                                                  contrast=-ratio
                                                                  )
            else:
                phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords,
                                                                  seed=1,
                                                                  parameter=number_of_pixels[0],
                                                                  contrast=1 / 10 ** ratio
                                                                  )

            if geometry_ID == 'sine_wave_' or geometry_ID == 'left_cluster_x3' or geometry_ID == 'right_cluster_x3' or geometry_ID == 'linear':
                if geometry_ID == 'sine_wave_' and ratio == 1:
                    pass
                else:
                    phase_field += 1 / 10 ** ratio
                    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
            # phase_fied_small_grid=np.copy(phase_field_smooth)
            phase_field = np.copy(phase_field)

            x = np.arange(0+0.5, 1 * number_of_pixels[0]+0.5)
            y = np.arange(0+0.5, 1 * number_of_pixels[1]+0.5)
            X_, Y_ = np.meshgrid(x, y)


            ax0 = fig.add_subplot(gs[G_a,2- T_a])
            pcm=ax0.pcolormesh(X_, Y_, np.transpose(phase_field),
                           cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0, alpha=0.5,
                           rasterized=True)
            # ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
            #           linestyle=linestyles[counter], linewidth=1.)
            # ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
            extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
            # extended_y = np.append(np.diag(phase_field), np.diag(phase_field)[-1])
            extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                                   phase_field[:, phase_field.shape[0] // 2][-1])

            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_aspect('equal')


            nb_disc = nb_pix_multips[T_a]
            # print(f'nb_pix_multip = {nb_pix_multip}')
            # system set up
            nb_discrete = (2 ** nb_disc, 2 ** nb_disc)


            triangles, X, Y = get_triangle(nx=nb_discrete[0], ny=nb_discrete[1]
                                           ,lx=number_of_pixels[0], ly=number_of_pixels[1])
            # Create the triangulation object
            triangulation = tri.Triangulation(X, Y, triangles)

            # Plot the mesh
            ax0.triplot(triangulation, 'k-', lw=0.1)
            #plt.scatter(X, Y, color='red', s=10)  # Plot nodes
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.title("Regular FEM Mesh")
            # plt.gca().set_aspect('equal')  # Keep aspect ratio
            # plt.show()
            if G_a == 2:
                ax0.set_xlabel( r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** nb_disc}}}$')
            if T_a == 0:
                ax0.set_ylabel(r'Geometry - ' + r'$\mathcal{G}$' + f'$_{{{2 ** nb_pix_multip}}}$')
                ax0.yaxis.set_label_position('right')
            counter += 1
    ax_cbar = fig.add_axes([0.14, 0.5, 0.02, 0.3])
    # 0.16, 0.22,
    cbar = plt.colorbar(pcm, location='right', cax=ax_cbar)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([r'$\frac{1}{\chi^{\rm tot}}$', 0.5, 1])
    ax_cbar.set_ylabel(r'Density $\rho_{\rm{cos} } $')

    ax_cbar.text(-0.25, 1.15, rf'\textbf{{(b)}}', transform=ax_cbar.transAxes)

    fname = src + 'JG_exp4_geometry_{}{}'.format(geometry_ID , '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

    plt.show()

    triangles, X, Y = get_triangle(nx=5, ny=5)
    # Create the triangulation object
    triangulation = tri.Triangulation(X, Y, triangles)

    # Plot the mesh
    plt.figure(figsize=(6, 6))
    plt.triplot(triangulation, 'k-', lw=0.8)
    plt.scatter(X, Y, color='red', s=10)  # Plot nodes
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regular FEM Mesh")
    plt.gca().set_aspect('equal')  # Keep aspect ratio
    plt.show()

    #
    nb_pix_multips = [4, 3, 2]
    geometry_ID = 'n_laminate'  # 'sine_wave_' n_laminate  linear  # 'abs_val' sine_wave_   ,laminate_log  geometry_ID = 'right_cluster_x3'  # laminate2       # 'abs_val' sine_wave_   ,laminate_log

    counter = 0
    ratio = 1
    fig = plt.figure(figsize=(4.25,4.25))
    gs = fig.add_gridspec(3, 3, hspace=0.1, wspace=0.1, width_ratios=[1, 1, 1],
                          height_ratios=[1, 1, 1])
    for G_a in np.arange(np.size(nb_pix_multips)):

        for T_a in np.arange(np.size(nb_pix_multips[0:G_a + 1])):
            print(f'Ga={G_a}')
            print(f'Ta={T_a}')
            nb_pix_multip = nb_pix_multips[G_a]
            # print(f'nb_pix_multip = {nb_pix_multip}')
            # system set up
            number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)

            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            # if kk == 0:
            if geometry_ID == 'laminate_log' or geometry_ID == 'n_laminate':
                phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords,
                                                                  seed=1,
                                                                  parameter=number_of_pixels[0],
                                                                  contrast=-ratio
                                                                  )
            else:
                phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords,
                                                                  seed=1,
                                                                  parameter=number_of_pixels[0],
                                                                  contrast=1 / 10 ** ratio
                                                                  )

            if geometry_ID == 'sine_wave_' or geometry_ID == 'left_cluster_x3' or geometry_ID == 'right_cluster_x3' or geometry_ID == 'linear':
                if geometry_ID == 'sine_wave_' and ratio == 1:
                    pass
                else:
                    phase_field += 1 / 10 ** ratio
                    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
            # phase_fied_small_grid=np.copy(phase_field_smooth)
            phase_field = np.copy(phase_field)

            x = np.arange(0 + 0.5, 1 * number_of_pixels[0] + 0.5)
            y = np.arange(0 + 0.5, 1 * number_of_pixels[1] + 0.5)
            X_, Y_ = np.meshgrid(x, y)

            ax0 = fig.add_subplot(gs[G_a, 2 - T_a])
            pcm=ax0.pcolormesh(X_, Y_, np.transpose(phase_field),
                           cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0, alpha=0.5,
                           rasterized=True)
            # ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
            #           linestyle=linestyles[counter], linewidth=1.)
            # ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
            extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
            # extended_y = np.append(np.diag(phase_field), np.diag(phase_field)[-1])
            extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                                   phase_field[:, phase_field.shape[0] // 2][-1])

            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_aspect('equal')

            nb_disc = nb_pix_multips[T_a]
            # print(f'nb_pix_multip = {nb_pix_multip}')
            # system set up
            nb_discrete = (2 ** nb_disc, 2 ** nb_disc)

            triangles, X, Y = get_triangle(nx=nb_discrete[0], ny=nb_discrete[1]
                                           , lx=number_of_pixels[0], ly=number_of_pixels[1])
            # Create the triangulation object
            triangulation = tri.Triangulation(X, Y, triangles)

            # Plot the mesh
            ax0.triplot(triangulation, 'k-', lw=0.1)
            # plt.scatter(X, Y, color='red', s=10)  # Plot nodes
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.title("Regular FEM Mesh")
            # plt.gca().set_aspect('equal')  # Keep aspect ratio
            # plt.show()
            if G_a == 2:
                ax0.set_xlabel(r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** nb_disc}}}$')
            if T_a == 0:
                ax0.set_ylabel(r'Geometry - ' + r'$\mathcal{G}$' + f'$_{{{2 ** nb_pix_multip}}}$')
                ax0.yaxis.set_label_position('right')
            counter += 1

    ax_cbar = fig.add_axes([0.14, 0.50, 0.02, 0.3])
    # 0.16, 0.22,
    cbar = plt.colorbar(pcm, location='right', cax=ax_cbar)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([r'$\frac{1}{\chi^{\rm tot}}$', 0.5, 1])
    ax_cbar.set_ylabel(r'Density $\rho_{\rm{laminate} } $')

    ax_cbar.text(-0.25, 1.15, rf'\textbf{{(a)}}', transform=ax_cbar.transAxes)
    fname = src + 'JG_exp4_single_geometry_{}{}'.format(geometry_ID, '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

    plt.show()
    quit()
