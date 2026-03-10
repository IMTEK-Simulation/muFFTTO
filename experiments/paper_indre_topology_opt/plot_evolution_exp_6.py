import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
from matplotlib.animation import FFMpegWriter

# Plotting parameters
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
})
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Arial"

# Setup parameters (matching exp_paper_TO_exp_6_square.py default args)
eta = 0.01
poison_target = 0.1
preconditioner_type = "Green_Jacobi"
nb_pixels = 1024
random_init = False
cg_tol_exponent = 8
soft_phase_exponent = 5

# Paths
script_dir = os.path.dirname(os.path.realpath(__file__))
base_script_name = 'exp_paper_TO_exp_5_square'
script_name = f'{base_script_name}_random_{random_init}_N_{nb_pixels}_cgtol_{cg_tol_exponent}_soft_{soft_phase_exponent}'
data_folder_path = os.path.join(script_dir, 'exp_data', script_name)
movie_folder_path = os.path.join(script_dir, 'movies', script_name)

if not os.path.exists(movie_folder_path):
    os.makedirs(movie_folder_path)

def get_iteration_files():
    # Adjusted pattern based on exp_paper_TO_exp_6_square.py callback
    # file_data_name_it = f'_eta_{eta}' + f'_p_{poison_target}' + f'_iteration_{iterat}'
    pattern = os.path.join(data_folder_path, f'{preconditioner_type}_eta_{eta}_p_{poison_target}_iteration_*.npy')
    files = glob.glob(pattern)
    
    # Extract iteration numbers for sorting
    def get_iter(f):
        match = re.search(r'iteration_(\d+)\.npy', f)
        return int(match.group(1)) if match else -1
    
    files.sort(key=get_iter)
    return files

def plot_evolution():
    parser = argparse.ArgumentParser(description="Plot evolution of phase field from exp_6")
    parser.add_argument("-t", "--tiles", type=int, default=1, help="Number of tiles in each direction")
    parser.add_argument("-fps", "--fps", type=int, default=10, help="Frames per second for the movie")
    args = parser.parse_args()
    nb_tiles = args.tiles
    fps = args.fps

    print(f"Searching for files in {data_folder_path}...")
    files = get_iteration_files()
    
    if not files:
        print(f"No files found matching the pattern.")
        return

    print(f"Found {len(files)} files. Creating movie...")

    # Load first file to get dimensions
    first_pf = np.load(files[0], allow_pickle=True)
    pf_tiled = np.tile(first_pf, (nb_tiles, nb_tiles))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    metadata = dict(title='Phase Field Evolution', artist='Matplotlib', comment='Phase field movie')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    movie_name = os.path.join(movie_folder_path, f'phase_field_evolution_tiles_{nb_tiles}.mp4')
    
    # Initialize plot with first frame
    im = ax.imshow(pf_tiled, cmap='Greys', origin='lower', vmin=0, vmax=1)
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
            
            # Update the image data
            im.set_data(pf_tiled)
            
            iteration_match = re.search(r"iteration_(\d+)", file_path)
            iteration_num = iteration_match.group(1) if iteration_match else "unknown"
            title.set_text(f'Iteration: {iteration_num}')
            
            writer.grab_frame()
    
    plt.close(fig)
    print(f"Movie saved to {movie_name}")

if __name__ == "__main__":
    plot_evolution()
