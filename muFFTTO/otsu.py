"""
Otsu-threshold based segmentation of 2D grayscale images.

The public function :func:`otsu_edgeDetection_and_phaseIndicator` turns a
grayscale image (e.g. a micrograph or a phase field) into

- a binary phase indicator (dark regions = 1),
- a one-pixel-wide mask of the external region boundaries (edges), and
- optionally a connected-component label field (one integer ID per region).

These masks are used, e.g., by ``muFFTTO.grid_adaptation_arbitrary`` to decide
where grid nodes should be moved onto material interfaces.

Otsu's method
-------------
For a histogram of intensities ``t in {0, ..., 255}`` Otsu's method chooses the
threshold ``t*`` that splits the pixels into two classes (below / above ``t``)
with maximal *between-class variance*::

    sigma_B^2(t) = w_0(t) w_1(t) (mu_0(t) - mu_1(t))^2,

where ``w_0, w_1`` are the class probabilities (pixel fractions) and
``mu_0, mu_1`` the class mean intensities. Maximizing ``sigma_B^2`` is
equivalent to minimizing the within-class variance. Here the actual
computation is delegated to OpenCV (``cv2.THRESH_OTSU``).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _normalize_to_u8(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a 2D image to [0, 1] and uint8 [0, 255].

    Min-max normalization ``(data - min) / (max - min)``; OpenCV's Otsu
    thresholding requires an 8-bit single-channel image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image of shape ``(nx, ny)``, any real dtype.

    Returns
    -------
    image_norm : numpy.ndarray
        float32 array of shape ``(nx, ny)`` with values in ``[0, 1]``. If the
        image is (numerically) constant, an all-zero array is returned.
    image_u8 : numpy.ndarray
        uint8 array of shape ``(nx, ny)``, ``image_norm * 255`` truncated
        (not rounded) to integers in ``[0, 255]``.

    Raises
    ------
    ValueError
        If ``data`` is not two-dimensional.
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

    data_min = float(data.min())
    data_max = float(data.max())
    value_range = data_max - data_min

    # guard against division by zero for constant images
    if value_range <= np.finfo(np.float32).eps:
        image_norm = np.zeros_like(data, dtype=np.float32)
    else:
        image_norm = (data - data_min) / value_range

    image_u8 = np.clip(image_norm * 255.0, 0, 255).astype(np.uint8)
    return image_norm, image_u8


def _validate_odd_positive(name: str, value: int) -> None:
    """Validate a positive odd OpenCV kernel size.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : int
        Kernel size to check.

    Raises
    ------
    ValueError
        If ``value`` is not a positive odd integer.
    """
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")


def _wrap_pad(image: np.ndarray, pad: int) -> np.ndarray:
    """Pad a 2D image periodically (wrap mode) by ``pad`` pixels on every side.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional image of shape ``(nx, ny)``.
    pad : int
        Non-negative pad width, smaller than ``min(nx, ny)``.

    Returns
    -------
    numpy.ndarray
        Padded image of shape ``(nx + 2 * pad, ny + 2 * pad)``.
    """
    if pad < 0 or pad >= min(image.shape):
        raise ValueError(
            f"[_wrap_pad] pad must satisfy 0 <= pad < min(image.shape)={min(image.shape)}, got {pad}."
        )
    if pad == 0:
        return image.copy()
    # PERIODIC: the image is treated as one periodic unit cell
    return np.pad(image, pad, mode="wrap")


def _crop(image: np.ndarray, pad: int) -> np.ndarray:
    """Remove a border of width ``pad`` added by :func:`_wrap_pad`."""
    if pad == 0:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(image[pad:-pad, pad:-pad])


def _periodic_edge_mask(phase_mask_binary: np.ndarray) -> np.ndarray:
    """One-pixel-wide periodic boundary mask of the foreground.

    A pixel is an edge pixel if it is foreground and at least one of its four
    periodic neighbours is background. This is the border-pixel definition of
    ``cv2.findContours`` (8-connected foreground, 4-connected background), but
    evaluated periodically, so no artificial edges appear on the image boundary.
    Boundaries of holes inside the foreground are included.

    Parameters
    ----------
    phase_mask_binary : numpy.ndarray
        Binary mask of shape ``(nx, ny)`` with values 0 (background) and 1.

    Returns
    -------
    numpy.ndarray
        uint8 edge mask of shape ``(nx, ny)`` with values 0 and 1.
    """
    fg = np.asarray(phase_mask_binary) > 0
    if fg.ndim != 2:
        raise ValueError(f"[_periodic_edge_mask] expected a 2D mask, got shape {fg.shape}.")
    # PERIODIC: neighbours across the image boundary via np.roll
    bg_neighbour = (
        ~np.roll(fg, 1, axis=0) | ~np.roll(fg, -1, axis=0)
        | ~np.roll(fg, 1, axis=1) | ~np.roll(fg, -1, axis=1)
    )
    return (fg & bg_neighbour).astype(np.uint8)


def _merge_periodic_labels(labels: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Merge connected-component labels that touch across the periodic boundary.

    Labels of foreground pixels on opposite image edges that are neighbours in
    the periodic sense (including diagonal neighbours for ``connectivity=8``)
    are merged with a union-find structure and renumbered to ``0, 1, ..., N``.
    Background stays label 0.

    Parameters
    ----------
    labels : numpy.ndarray
        int32 label image of shape ``(nx, ny)`` from cv2 connected components.
    connectivity : int
        4 or 8, the connectivity used for the labeling.

    Returns
    -------
    labels_periodic : numpy.ndarray
        int32 label image of shape ``(nx, ny)``.
    number_of_regions : int
        Number of foreground regions (background excluded).
    """
    labels = np.asarray(labels, dtype=np.int32)
    n_labels = int(labels.max()) + 1
    parent = list(range(n_labels))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    shifts = (0,) if connectivity == 4 else (-1, 0, 1)
    pairs = []
    # PERIODIC: last row touches first row, last column touches first column
    for s in shifts:
        pairs.append(np.stack((labels[-1, :], np.roll(labels[0, :], s)), axis=1))
        pairs.append(np.stack((labels[:, -1], np.roll(labels[:, 0], s)), axis=1))
    pairs = np.concatenate(pairs)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    for a, b in np.unique(pairs, axis=0):
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    roots = np.array([find(a) for a in range(n_labels)], dtype=np.int32)
    lookup = np.zeros(n_labels, dtype=np.int32)
    unique_roots = np.unique(roots[1:]) if n_labels > 1 else np.empty(0, dtype=np.int32)
    for new_id, root in enumerate(unique_roots, start=1):
        lookup[roots == root] = new_id
    lookup[0] = 0
    return lookup[labels], int(unique_roots.size)


def otsu_edgeDetection_and_phaseIndicator(
        data: np.ndarray,
        blur_ksize: int = 7,
        blur_sigma: float = 1.5,
        morph_kernel_size: int = 3,
        open_iterations: int = 1,
        close_iterations: int = 1,
        regions_label: bool = True,
        connectivity: int = 8,
        return_intermediate: bool = False,
        periodic: bool = True,

) -> dict[str, Any]:
    """
    Segment dark regions in a two-dimensional grayscale image, extract their
    external boundaries, and optionally assign a unique integer label to each
    disconnected foreground region.

    The input image is normalized to uint8 intensity values, smoothed with a
    Gaussian filter, and segmented using inverted Otsu thresholding. Morphological
    opening and closing are then applied to reduce isolated noise, remove small
    foreground artifacts, fill small holes, and close short gaps.

    Inverted Otsu thresholding treats dark pixels as foreground:

    - ``phase_mask_binary == 1`` identifies dark segmented regions.
    - ``phase_mask_binary == 0`` identifies bright/background regions.

    When ``regions_label=True``, connected-component analysis assigns one unique
    integer ID to each disconnected foreground region. Background pixels always
    have label ``0``; foreground region labels are consecutive integers
    ``1, 2, ..., N``.

    This function currently distinguishes regions by geometric connectivity only.
    Therefore, two disconnected regions receive different labels even when their
    average grayscale values are similar. A later processing step may group such
    regions into shared material classes based on their mean grayscale values.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional grayscale input image with shape ``(nx, ny)``. Integer and
        floating-point input dtypes are supported.

    blur_ksize : int, default=7
        Positive odd Gaussian-kernel width and height. Larger values produce
        stronger smoothing before Otsu thresholding.

    blur_sigma : float, default=1.5
        Standard deviation of the Gaussian filter. Must be non-negative.

    morph_kernel_size : int, default=3
        Positive odd side length of the square structuring element used for
        morphological opening and closing.

    open_iterations : int, default=1
        Number of morphological-opening iterations. Opening suppresses small
        isolated foreground pixels and thin noise.

    close_iterations : int, default=1
        Number of morphological-closing iterations. Closing fills small holes and
        connects short gaps inside foreground regions.

    regions_label : bool, default=True
        If ``True``, perform connected-component labeling on the cleaned binary
        foreground mask.

        - ``True``: ``phase_mask_label`` has dtype ``int32`` and contains labels
          ``0, 1, ..., N``.
        - ``False``: ``phase_mask_label`` is a uint8 copy of
          ``phase_mask_binary`` and contains only ``0`` and ``1``.

    connectivity : int, default=8
        Connectivity used for connected-component labeling when
        ``regions_label=True``.

        - ``4`` connects pixels sharing a horizontal or vertical edge.
        - ``8`` additionally connects pixels touching diagonally at a corner.

    return_intermediate : bool, default=False
        If ``True``, include normalized, blurred, thresholded, and morphology
        intermediate arrays in the returned dictionary.

    periodic : bool, default=True
        Treat the image as one periodic unit cell.

        - Gaussian blur and morphology use periodic (wrap) padding.
        - The Otsu threshold is computed on the unpadded image only.
        - ``edge_mask`` is the periodic boundary of the foreground; no
          artificial edges appear on the image boundary, and boundaries of
          holes inside the foreground (background enclosed by foreground)
          are included.
        - Region labels touching across the periodic boundary are merged.
        - ``phase_region_stats`` and ``phase_region_centroids`` (bounding boxes
          and centroids are undefined for wrapped regions) are replaced by
          ``phase_region_areas``.

        ``periodic=False`` reproduces the original non-periodic behaviour.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the following arrays with shape ``(nx, ny)``:

        ``edge_mask`` : numpy.ndarray, uint8
            One-pixel-wide mask of external contours. A value of ``1`` marks an
            outer boundary of a segmented foreground region.

        ``phase_mask_binary`` : numpy.ndarray, uint8
            Cleaned binary segmentation mask with values ``0`` for background and
            ``1`` for dark foreground regions.

        ``phase_mask_label`` : numpy.ndarray
            Region-label field.

            - If ``regions_label=True``, dtype is ``int32`` and values are
              ``0, 1, ..., N``.
            - If ``regions_label=False``, dtype is ``uint8`` and values are
              ``0`` and ``1``.

        When ``regions_label=True`` and ``periodic=True``, the dictionary
        additionally contains ``number_of_phase_regions`` (int) and
        ``phase_region_areas`` (numpy.ndarray of shape ``(N + 1,)``, pixel
        count of every label including background label ``0``).

        When ``regions_label=True`` and ``periodic=False``, the dictionary
        additionally contains:

        ``number_of_phase_regions`` : int
            Number of disconnected foreground regions. Background label ``0`` is
            excluded.

        ``phase_region_stats`` : numpy.ndarray
            OpenCV connected-component statistics. Each row corresponds to one
            region label, including background label ``0``. The columns contain
            bounding-box x/y coordinates, width, height, and pixel area.

        ``phase_region_centroids`` : numpy.ndarray
            Centroid coordinates ``(x, y)`` for every label, including background.

        When ``return_intermediate=True``, the dictionary additionally contains:

        ``image_norm`` : numpy.ndarray
            Input image normalized to the interval ``[0, 1]``.

        ``image_u8`` : numpy.ndarray
            Normalized uint8 image with intensity values in ``[0, 255]``.

        ``image_blur`` : numpy.ndarray
            Gaussian-smoothed uint8 image.

        ``mask_raw`` : numpy.ndarray, uint8
            Binary result directly after inverted Otsu thresholding.

        ``mask_open`` : numpy.ndarray, uint8
            Binary result after morphological opening.

        ``mask_clean`` : numpy.ndarray, uint8
            Binary result after opening and closing; equal to
            ``phase_mask_binary``.

        ``otsu_threshold`` : float
            Automatically selected Otsu threshold in uint8 intensity units.

        ``contour_count`` : int or None
            Number of external contours detected from ``phase_mask_binary``
            (``None`` when ``periodic=True``; contours are not used then).

    Notes
    -----
    With ``periodic=False`` the function applies standard non-periodic image
    connectivity, so a region crossing the periodic boundary receives separate
    labels at opposite image edges and ``edge_mask`` contains artificial edges
    on the image boundary. Use ``periodic=True`` (default) for periodic unit
    cells and periodic grid adaptation.

    The returned ``phase_mask_label`` describes disconnected geometric regions,
    not material classes. In a future extension, regions with similar mean
    grayscale values can be grouped into a separate ``material_label`` field,
    while retaining ``phase_mask_label`` for region geometry.
    """

    _validate_odd_positive("blur_ksize", blur_ksize)
    _validate_odd_positive("morph_kernel_size", morph_kernel_size)

    if blur_sigma < 0:
        raise ValueError(f"blur_sigma must be non-negative, got {blur_sigma}.")
    if open_iterations < 0:
        raise ValueError(
            f"open_iterations must be non-negative, got {open_iterations}."
        )
    if close_iterations < 0:
        raise ValueError(
            f"close_iterations must be non-negative, got {close_iterations}."
        )
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity}."
        )

    # (1) min-max normalization to [0, 1] and 8-bit [0, 255] (required by OpenCV Otsu)
    image_norm, image_u8 = _normalize_to_u8(data)

    # Pad widths for periodic processing: the blur needs blur_ksize // 2,
    # opening + closing spread by 2 * radius per iteration each.
    morph_radius = morph_kernel_size // 2
    pad_blur = blur_ksize // 2 if periodic else 0
    pad_morph = 2 * morph_radius * (open_iterations + close_iterations) + 1 if periodic else 0

    # (2) Gaussian smoothing suppresses pixel noise so that the histogram is
    #     closer to bimodal and the thresholded mask is less ragged
    # PERIODIC: blur a wrap-padded image and crop back to the original size
    image_blur = _crop(
        cv2.GaussianBlur(
            _wrap_pad(image_u8, pad_blur),
            (blur_ksize, blur_ksize),
            blur_sigma,
        ),
        pad_blur,
    )

    # (3) Otsu thresholding: the passed threshold value 0 is ignored, OpenCV
    #     selects t* maximizing the between-class variance of the histogram.
    #     THRESH_BINARY_INV -> pixels with intensity <= t* (dark) become 255,
    #     brighter pixels become 0.
    otsu_threshold, mask_raw_u8 = cv2.threshold(
        image_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # (4) morphological cleanup with a square structuring element:
    #     opening  = erosion followed by dilation -> removes small specks
    #     closing  = dilation followed by erosion -> fills small holes / gaps
    kernel = np.ones(
        (morph_kernel_size, morph_kernel_size),
        dtype=np.uint8,
    )

    # PERIODIC: opening and closing on a wrap-padded mask, cropped afterwards
    mask_open_pad_u8 = cv2.morphologyEx(
        _wrap_pad(mask_raw_u8, pad_morph),
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations,
    )

    mask_clean_pad_u8 = cv2.morphologyEx(
        mask_open_pad_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iterations,
    )
    mask_open_u8 = _crop(mask_open_pad_u8, pad_morph)
    mask_clean_u8 = _crop(mask_clean_pad_u8, pad_morph)

    # Binary phase indicator: values are always 0 or 1.
    phase_mask_binary = (mask_clean_u8 > 0).astype(np.uint8)

    if periodic:
        # PERIODIC: boundary pixels evaluated with periodic neighbours
        # (includes boundaries of holes, no artificial image-border edges).
        contours = None
        edge_mask = _periodic_edge_mask(phase_mask_binary)
    else:
        # Detect edges from the binary phase mask.
        # RETR_EXTERNAL: only outermost contours (holes inside regions are ignored);
        # CHAIN_APPROX_SIMPLE compresses straight segments, but drawContours below
        # re-draws them as continuous 1-pixel-wide polylines.
        contours, _ = cv2.findContours(
            (phase_mask_binary * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        edge_mask = np.zeros_like(phase_mask_binary, dtype=np.uint8)

        cv2.drawContours(
            edge_mask,
            contours,
            contourIdx=-1,
            color=1,
            thickness=1,
        )

    # Default behavior: label field is the same as the binary mask.
    phase_mask_label = phase_mask_binary.copy()

    # Legacy/alternative implementation kept for reference (inactive string
    # literal, never executed). English summary of the plan described below:
    #   (1) label connected foreground regions with connectedComponentsWithStats
    #       (unique ID per region, label 0 = background);
    #   (2) compute the mean gray value of every region from image_norm;
    #   (3) give regions with sufficiently similar mean gray value the same
    #       material_label -> separates "geometric regions" from "materials".
    #   Planned outputs: phase_mask_binary (0/1), phase_mask_label (0..N),
    #   material_label (0..M), region_mean_gray, region_to_material.
    #   Inline notes: each material group stores [representative mean gray,
    #   number of regions]; regions are sorted by mean gray so that the label
    #   numbering does not influence the grouping; the group mean is updated as
    #   a running average; a new material class is created if no group is
    #   within gray_tolerance.
    '''
    (1) connectedComponentsWithStats 對 binary foreground 做連通區域標記，得到每個 region 的唯一 ID。
    (2) 對每個 region，從原始 normalized grayscale image計算平均灰階。
    (3) 比較各 region 的平均灰階；平均值足夠相近的 region，給予相同 material_label。
     --> 會同時得到「每一塊幾何區域」與「每一種材料」兩種不同資料。connectedComponentsWithStats 可輸出 label map、每個區域統計與中心位置，而 label 0 是背景

    [output]:
    phase_mask_binary : 0/1，Otsu 分割結果
    phase_mask_label  : 0/1/2/3/...，每個連通區域有唯一 ID
    material_label    : 0/1/2/...，平均灰階相近的區域共享材料 ID
    region_mean_gray  : 每個 region 的平均灰階
    region_to_material: region ID -> material ID 的對照表

    def _validate_gray_tolerance(gray_tolerance: float) -> None:
    """Validate the tolerance used to group region mean grayscale values."""
    if gray_tolerance < 0.0:
        raise ValueError(
            "gray_tolerance must be non-negative, "
            f"got {gray_tolerance}."
        )


    def _group_regions_by_mean_gray(
        image_norm: np.ndarray,
        phase_mask_label: np.ndarray,
        gray_tolerance: float,
    ) -> tuple[np.ndarray, dict[int, float], dict[int, int]]:
        """
        Assign material IDs by grouping connected regions with similar mean gray.

        Parameters
        ----------
        image_norm
            Original image normalized to [0, 1].
        phase_mask_label
            Connected-region label map. Label 0 is background.
        gray_tolerance
            Two regions are assigned to the same material when the absolute
            difference between their representative mean gray values is not
            greater than this tolerance.

        Returns
        -------
        material_label
            Integer material-ID field. Background remains 0. Foreground material
            IDs start at 1.

        region_mean_gray
            Dictionary mapping region ID to its mean normalized grayscale value.

        region_to_material
            Dictionary mapping region ID to its assigned material ID.
        """
        _validate_gray_tolerance(gray_tolerance)

        labels = np.asarray(phase_mask_label, dtype=np.int32)
        material_label = np.zeros_like(labels, dtype=np.int32)

        region_ids = np.unique(labels)
        region_ids = region_ids[region_ids != 0]

        region_mean_gray: dict[int, float] = {}

        for region_id in region_ids:
            region_id = int(region_id)
            region_pixels = image_norm[labels == region_id]

            if region_pixels.size == 0:
                continue

            region_mean_gray[region_id] = float(region_pixels.mean())

        # 每一組儲存：
        # [代表平均灰階, 此組包含幾個 region]
        material_groups: list[list[float]] = []

        region_to_material: dict[int, int] = {}

        # 先依平均灰階排序，避免 label 的編號順序影響分類結果。
        sorted_region_ids = sorted(
            region_mean_gray,
            key=lambda region_id: region_mean_gray[region_id],
        )

        for region_id in sorted_region_ids:
            region_gray = region_mean_gray[region_id]
            material_id: int | None = None

            for group_index, group in enumerate(material_groups):
                group_mean_gray = group[0]
                group_region_count = group[1]

                if abs(region_gray - group_mean_gray) <= gray_tolerance:
                    material_id = group_index + 1

                    # 以目前 group 成員的平均值更新代表灰階。
                    group[0] = (
                        group_mean_gray * group_region_count + region_gray
                    ) / (group_region_count + 1)

                    group[1] = group_region_count + 1
                    break

            # 找不到相近 group，建立一個新材料類別。
            if material_id is None:
                material_groups.append([region_gray, 1.0])
                material_id = len(material_groups)

            region_to_material[region_id] = material_id
            material_label[labels == region_id] = material_id

        return material_label, region_mean_gray, region_to_material
    '''
    # Optional: assign one integer label to each connected region.
    # (non-periodic connectivity, see Notes in the docstring). Returns the
    # number of labels including background, the int32 label image, the stats
    # array of shape (number_of_labels, 5) with columns
    # [x, y, width, height, area] and the centroids of shape (number_of_labels, 2).
    if regions_label:
        number_of_labels, phase_mask_label, phase_region_stats, phase_region_centroids = (
            cv2.connectedComponentsWithStats(
                phase_mask_binary,
                connectivity=connectivity,
                ltype=cv2.CV_32S,
            )
        )
        if periodic:
            # PERIODIC: merge regions that continue across the image boundary
            phase_mask_label, number_of_regions = _merge_periodic_labels(
                phase_mask_label, connectivity
            )
            phase_region_areas = np.bincount(
                phase_mask_label.ravel(), minlength=number_of_regions + 1
            )

    results: dict[str, Any] = {
        "edge_mask": edge_mask,
        "phase_mask_binary": phase_mask_binary,
        "phase_mask_label": phase_mask_label,
    }

    # These data exist only when connected-component labeling was requested.
    if regions_label and periodic:
        results.update(
            {
                "number_of_phase_regions": int(number_of_regions),
                "phase_region_areas": phase_region_areas,
            }
        )
    elif regions_label:
        results.update(
            {
                "number_of_phase_regions": int(number_of_labels - 1),
                "phase_region_stats": phase_region_stats,
                "phase_region_centroids": phase_region_centroids,
            }
        )

    if return_intermediate:
        results.update(
            {
                "image_norm": image_norm,
                "image_u8": image_u8,
                "image_blur": image_blur,
                "mask_raw": (mask_raw_u8 > 0).astype(np.uint8),
                "mask_open": (mask_open_u8 > 0).astype(np.uint8),
                "mask_clean": phase_mask_binary,
                "otsu_threshold": float(otsu_threshold),
                "contour_count": None if contours is None else int(len(contours)),
            }
        )

    return results
