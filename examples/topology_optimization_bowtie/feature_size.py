"""
Minimum feature size of a periodic 0/1 design (solid members and voids), measured by morphological opening.

The design is thresholded at 0.5. A disk of diameter d (pixels) "fits" the solid if the opening of the solid with
the disk (erosion followed by dilation: the union of all disks of diameter d inside the solid)
  * only trims the boundary: every removed pixel lies within max(1.5, 0.2 d) pixels of the opened solid (rounding
    of convex corners and of pixel steps), while a thin bar or hinge leaves removed pixels far from the rest, and
  * keeps the number of connected solid pieces (a cut hinge splits the structure, a thin piece disappears).
The minimum solid feature size is the largest d that fits; the minimum void size is the same for the void (opening
of the void = closing of the solid). Periodicity: the design is tiled 3 x 3 and only the centre cell is evaluated.

    python feature_size.py data/<run> [data/<run> ...]       # prints the sizes of the final designs
"""
import os
import sys

import numpy as np
from scipy import ndimage


def disk(diameter):
    """Boolean disk of the given diameter in pixels (odd or even), centred in its bounding box."""
    radius = diameter / 2.0
    n = int(np.ceil(diameter))
    centre = (n - 1) / 2.0
    i, j = np.mgrid[:n, :n]
    return (i - centre) ** 2 + (j - centre) ** 2 <= radius ** 2 + 1e-9


def _centre(tiled, shape):
    return tiled[shape[0]:2 * shape[0], shape[1]:2 * shape[1]]


def _pieces(mask):
    """Number of connected pieces of a periodic mask (8-connectivity), counted on the 3 x 3 tiling."""
    labels, _ = ndimage.label(np.tile(mask, (3, 3)), structure=np.ones((3, 3)))
    return len(np.unique(_centre(labels, mask.shape)[_centre(labels, mask.shape) > 0]))


def fits(mask, diameter):
    """True if a disk of ``diameter`` pixels fits the periodic phase ``mask`` (see the module docstring)."""
    if not mask.any():
        return True
    opened_tiled = ndimage.binary_opening(np.tile(mask, (3, 3)), structure=disk(diameter))
    if not opened_tiled.any():
        return False
    distance_to_opened = _centre(ndimage.distance_transform_edt(~opened_tiled), mask.shape)
    removed = mask & ~_centre(opened_tiled, mask.shape)
    only_trimmed = not removed.any() or distance_to_opened[removed].max() <= max(1.5, 0.2 * diameter)
    return only_trimmed and _pieces(_centre(opened_tiled, mask.shape)) == _pieces(mask)


def minimum_feature_size(mask, max_diameter=None):
    """Largest disk diameter (pixels, steps of 1) that fits the phase ``mask``; 0 if not even one pixel fits."""
    max_diameter = max_diameter or min(mask.shape)
    size = 0
    for diameter in range(1, max_diameter + 1):
        if not fits(mask, diameter):
            break
        size = diameter
    return size


def feature_sizes(rho, threshold=0.5):
    """(minimum solid size, minimum void size) in pixels of the design ``rho`` thresholded at ``threshold``."""
    solid = np.asarray(rho) > threshold
    return minimum_feature_size(solid), minimum_feature_size(~solid)


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from muFFTTO import io_utils
    for folder in sys.argv[1:]:
        data = io_utils.read_file(os.path.join(folder, 'final.nc'))
        rho = np.asarray(data.fields['rho_final']).reshape(np.asarray(data.attributes['nb_of_pixels']).astype(int))
        solid, void = feature_sizes(rho)
        print(f'{os.path.basename(folder)}: minimum solid {solid} px, minimum void {void} px')
