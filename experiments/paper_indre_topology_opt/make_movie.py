import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re
from matplotlib.animation import FFMpegWriter

# Plotting parameters
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
})
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Arial"

# Setup parameters (matching exp_paper_TO_exp_4_hexa_w_dep_plots.py)
N = 1024
nb_tiles = 3
random_init = False
cg_tol_exponent = 8
soft_phase_exponent = 5
eta_mult = 0.01

# Downsampling for faster plotting
DOWNSAMPLE = N > 512
if DOWNSAMPLE:
    step = N // 512
else:
    step = 1

#weights = np.array([0.1,  0.3,  0.7,  1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0])
weights = np.array([ 10.0, 30.0, 70.0, 100.0])

# Paths
grid= 'square'#'hexa' #
experiment=2
script_dir = os.path.dirname(os.path.realpath(__file__))
script_name = f'exp_paper_TO_exp_{experiment}_{grid}_random_{random_init}_N_{N}_cgtol_{cg_tol_exponent}_soft_{soft_phase_exponent}'
data_folder_path = os.path.join(script_dir, 'exp_data', script_name)
movie_folder_path = os.path.join(script_dir, 'movies', script_name)

if not os.path.exists(movie_folder_path):
    os.makedirs(movie_folder_path)

preconditioner_type = "Green_Jacobi"

# Tiling logic
x_ref = np.zeros([2, nb_tiles * N + 1, nb_tiles * N + 1])
x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * N + 1),
                                 np.linspace(0, nb_tiles, nb_tiles * N + 1), indexing='ij')
shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * N + 1)
x_coords = np.copy(x_ref)
x_coords[0] += shift[None, :] - 2
x_coords[1] *= np.sqrt(3) / 2

# Pre-downsample coordinates if needed
if DOWNSAMPLE:
    x_c0 = x_coords[0][::step, ::step]
    x_c1 = x_coords[1][::step, ::step]
else:
    x_c0 = x_coords[0]
    x_c1 = x_coords[1]

def get_iteration_files(weight):
    pattern = os.path.join(data_folder_path, f'{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_iteration_*.npy')
    files = glob.glob(pattern)
    
    # Extract iteration numbers for sorting
    def get_iter(f):
        match = re.search(r'iteration_(\d+)\.npy', f)
        return int(match.group(1)) if match else -1
    
    files.sort(key=get_iter)
    return files

for w_mult in weights:
    print(f"Processing weight: {w_mult}")
    files = get_iteration_files(w_mult)
    
    if not files:
        print(f"No files found for weight {w_mult}")
        continue

    fig, ax = plt.subplots(figsize=(8, 8))
    
    metadata = dict(title=f'Phase Field Evolution w={w_mult}', artist='Matplotlib', comment='Phase field movie')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    movie_name = os.path.join(movie_folder_path, f'movie_w_{w_mult:.1f}.mp4')
    
    # Initialize plot with first frame
    try:
        phase_field = np.load(files[0], allow_pickle=True)
    except Exception as e:
        print(f"Error loading {files[0]}: {e}")
        plt.close(fig)
        continue

    pf_tiled = np.tile(phase_field, (nb_tiles, nb_tiles))
    if DOWNSAMPLE:
        pf_display = pf_tiled[::step, ::step]
    else:
        # Padding to match x_coords if necessary (N*nb_tiles + 1)
        pf_display = np.zeros((nb_tiles * N + 1, nb_tiles * N + 1))
        pf_display[:-1, :-1] = pf_tiled
        pf_display[-1, :] = pf_display[0, :]
        pf_display[:, -1] = pf_display[:, 0]

    if grid == 'square':
        im = ax.pcolormesh(pf_display, cmap='Greys', shading='auto', rasterized=True)
    else:
        im = ax.pcolormesh(x_c0, x_c1, pf_display, cmap='Greys', shading='auto', rasterized=True)
    
    ax.set_aspect('equal')
    ax.axis('off')
    title = ax.set_title('')
    
    with writer.saving(fig, movie_name, dpi=100):
        for i, file_path in enumerate(files):
            if i % 10 == 0:
                print(f"  Frame {i}/{len(files)}")
            
            try:
                phase_field = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            pf_tiled = np.tile(phase_field, (nb_tiles, nb_tiles))
            if DOWNSAMPLE:
                pf_display = pf_tiled[::step, ::step]
            else:
                pf_display = np.zeros((nb_tiles * N + 1, nb_tiles * N + 1))
                pf_display[:-1, :-1] = pf_tiled
                pf_display[-1, :] = pf_display[0, :]
                pf_display[:, -1] = pf_display[:, 0]

            # Update the image data
            # pcolormesh returns a QuadMesh. set_array expects a 1D array of values for the faces.
            # For shading='auto', it handles it correctly if we pass the right sized array.
            im.set_array(pf_display.ravel())
            
            iteration_match = re.search(r"iteration_(\d+)", file_path)
            iteration_num = iteration_match.group(1) if iteration_match else "unknown"
            title.set_text(f'Weight: {w_mult:.1f}, Iteration: {iteration_num}')
            
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Movie saved to {movie_name}")
