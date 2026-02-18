import numpy as np
from moviepy import VideoFileClip, ColorClip, clips_array

# Your weights
weights = np.array([0.1, 0.3, 0.7, 1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0])

# 1. Generate filenames from weights
files = [f"./movies/exp_paper_TO_exp_4_hexa_random_False_N_1024_cgtol_8_soft_5/movie_w_{w:.1f}.mp4" for w in weights]
print("Files:", files)

# 2. Load clips
clips = [VideoFileClip(f) for f in files]

# 3. Equalize duration (MoviePy requires same length for grids)
min_duration = min(c.duration for c in clips)
clips = [c.subclipped(0, min_duration) for c in clips]

# 4. Resize (adjust height as needed)
clips = [c.resized(height=240) for c in clips]

# 5. Fill to 12 clips for a 4×3 grid
while len(clips) < 10:
    black = ColorClip(size=clips[0].size, color=(0,0,0), duration=min_duration)
    clips.append(black)

# 6. Arrange into 4×3 grid
grid = clips_array([
    clips[0:5],
    clips[5:10]
])

# 7. Export
grid.write_videofile("combined_grid.mp4", fps=24)
