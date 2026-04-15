if MPI.COMM_WORLD.rank == 0:
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Select 3 load increments to plot
load_increments_to_plot = [0.1, 0.5, 1.0]  # Adjust these values as needed

for plot_idx, load_increment in enumerate(load_increments_to_plot):
    inc_index = int(load_increment * 10)

    file_data_name_it = f'_w_{weight}' + f'_p_{poison_target}' + f'_load_increment_{load_increment}' + f'_tilled{nb_tiles}'

    displacement = load_npy(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'.npy',
                            subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                            nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                            components_are_leading=True,
                            comm=MPI.COMM_WORLD)

    _info = np.load(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'_log_plotting.npz',
                    allow_pickle=True)
    macro_gradient = _info.f.macro_grads_corrected
    print(f'macro_gradient = {macro_gradient}')

    if MPI.COMM_WORLD.rank == 0:
        repetition = 3
        x_ref = np.zeros([2, repetition * (N) + 1, repetition * (N) + 1])
        x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, repetition, repetition * (N) + 1),
                                         np.linspace(0, repetition, repetition * (N) + 1), indexing='ij')
        shift = 0.5 * np.linspace(0, repetition, repetition * (N) + 1)

        x_coords = np.copy(x_ref)
        if grid_type == 'hex':
            x_coords[0] += shift[None, :]
            x_coords[1] *= np.sqrt(3) / 2

        lin_disp_ixy = np.einsum('ij...,j...->i...', macro_gradient, x_coords)

        x_coords[0] += lin_disp_ixy[0]
        x_coords[1] += lin_disp_ixy[1]

        # add fluctuation of
        x_coords[0, :-1, :-1] += np.tile(displacement[0], (repetition, repetition))
        x_coords[1, :-1, :-1] += np.tile(displacement[1], (repetition, repetition))

        # build a periodic displacement
        tilled_disp_x = np.tile(displacement[0], (repetition, repetition))
        tilled_disp_y = np.tile(displacement[1], (repetition, repetition))

        # Fill last row and column with first row and column
        x_coords[0, -1, :-1] += tilled_disp_x[0, :]
        x_coords[0, :-1, -1] += tilled_disp_x[:, 0]
        x_coords[0, -1, -1] += tilled_disp_x[0, 0]

        x_coords[1, -1, :-1] += tilled_disp_y[0, :]
        x_coords[1, :-1, -1] += tilled_disp_y[:, 0]
        x_coords[1, -1, -1] += tilled_disp_y[0, 0]

        ax = axes[plot_idx]
        pcm = ax.pcolormesh(x_coords[0], x_coords[1], np.tile(phase_field.s[0, 0], (repetition, repetition)),
                            shading='flat',
                            edgecolors='none',
                            lw=0.01,
                            cmap=mpl.cm.Greys,
                            vmin=0, vmax=1,
                            rasterized=True)

        if plot_idx == 2:  # Add colorbar only to the last plot
            fig.colorbar(pcm, ax=ax)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        if grid_type == 'hex':
            ax.set_xlim(-0.15, repetition + repetition * 0.8)
            ax.set_ylim(-0.15, repetition + repetition * 0.05)
        else:
            ax.set_xlim(-0.15, repetition + repetition * 0.2)
            ax.set_ylim(-0.15, repetition + repetition * 0.2)

        ax.set_title(f'Load increment {load_increment:.1f}')
        ax.set_aspect('equal')

        # Add grid lines
        for i in range(nb_tiles + 1):
            if grid_type == 'hex':
                y_val = i * np.sqrt(3) / 2
                ax.plot([0 - 2, nb_tiles + 0.5 * nb_tiles - 2], [y_val, y_val],
                        color='k', linestyle='--', linewidth=1, alpha=0.5)
                x_start = i - 2
                x_end = i + 0.5 * nb_tiles - 2
                ax.plot([x_start, x_end], [0, nb_tiles * np.sqrt(3) / 2],
                        color='k', linestyle='--', linewidth=1, alpha=0.5)
            else:
                ax.axhline(y=i, color='k', linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(x=i, color='k', linestyle='--', linewidth=1, alpha=0.5)

if MPI.COMM_WORLD.rank == 0:
    plt.tight_layout()
    figure_name = figure_folder_path + f'grid_{grid_type}_comparison_w_{weight}_p_{poison_target}_N{N}_{nb_tiles}.png'
    plt.savefig(figure_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {figure_name}')
