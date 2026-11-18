"""
GPU memory accounting for the large-3D-TO study (WP1 of FABLE_BRIEF.md): who owns the bytes of a run?

    owners = {'TO': discretization, 'MG level 0': level.disc, ...}
    fable_memory.report(owners, nb_voxels)

prints, per owner, every muGrid field (real and Fourier collections) with its size in MiB and bytes per fine voxel,
then the CuPy memory pool (live arrays by shape, and the reserved-but-free part = the high-water mark of temporaries),
and what is left of the process' device memory (CUDA context, cuFFT plans and work areas, muGrid ghost buffers).

Everything is read from the libraries: muGrid `Field` buffer sizes, `cupy.get_default_memory_pool()`, and
`nvidia-smi --query-compute-apps` for the process total.
"""
import gc
import os
import subprocess

import numpy as np

MiB = 2.0 ** 20


def field_bytes(collection):
    """{field name: bytes} of one muGrid field collection (ghost layers included)"""
    cpp = collection._cpp if hasattr(collection, '_cpp') else collection
    sizes = {}
    for name in cpp.field_names:
        field = cpp.get_field(name)
        sizes[name] = int(field.buffer_size) * int(field.element_size_in_bytes)
    return sizes


def process_gpu_bytes():
    """device memory of this process as nvidia-smi sees it (0 if not listed, e.g. on the CPU)"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in out.splitlines():
        pid, used = line.split(',')
        if int(pid) == os.getpid():
            return int(used) * 2 ** 20
    return 0


def live_cupy_arrays():
    """{(shape, dtype): [count, bytes]} of the CuPy arrays that own pool memory right now"""
    import cupy
    seen, groups = set(), {}
    for obj in gc.get_objects():
        if isinstance(obj, cupy.ndarray) and obj.data.mem is not None:
            mem = obj.data.mem
            if getattr(mem, 'ptr', 0) in seen or not isinstance(mem, cupy.cuda.memory.PooledMemory):
                continue
            seen.add(mem.ptr)
            entry = groups.setdefault((obj.shape, str(obj.dtype)), [0, 0])
            entry[0] += 1
            entry[1] += mem.size
    return groups


def report(owners, nb_voxels, min_bytes_per_voxel=4.0, print_fn=print):
    """Memory table of a run. owners: {label: Discretization}; nb_voxels: fine-grid voxels (the per-voxel unit).
    Returns {'fields': bytes, 'pool_used': bytes, 'pool_reserved': bytes, 'process': bytes}."""
    def line(label, nbytes):
        print_fn(f'MEM   {label:<62s} {nbytes / MiB:10.1f} MiB {nbytes / nb_voxels:9.1f} B/voxel')

    total_fields = 0
    for label, disc in owners.items():
        sizes = {}
        for kind, collection in (('', disc.field_collection), (' (Fourier)', disc.ffield_collection)):
            for name, nbytes in field_bytes(collection).items():
                sizes[name + kind] = nbytes
        subtotal = sum(sizes.values())
        total_fields += subtotal
        print_fn(f'MEM {label}: {len(sizes)} muGrid fields, grid {tuple(int(n) for n in disc.nb_of_pixels_global)}')
        small = 0
        for name, nbytes in sorted(sizes.items(), key=lambda item: -item[1]):
            if nbytes / nb_voxels >= min_bytes_per_voxel:
                line(name, nbytes)
            else:
                small += nbytes
        if small:
            line('(smaller fields)', small)
        line(f'= {label} total', subtotal)
    result = {'fields': total_fields, 'pool_used': 0, 'pool_reserved': 0, 'process': process_gpu_bytes()}
    try:
        import cupy
        pool = cupy.get_default_memory_pool()
        result['pool_used'], result['pool_reserved'] = pool.used_bytes(), pool.total_bytes()
        print_fn('MEM CuPy pool: live arrays by shape')
        for (shape, dtype), (count, nbytes) in sorted(live_cupy_arrays().items(), key=lambda item: -item[1][1]):
            if nbytes / nb_voxels >= min_bytes_per_voxel:
                line(f'{count} x {shape} {dtype}', nbytes)
        line('= CuPy pool in use', result['pool_used'])
        line('= CuPy pool reserved but free (peak of temporaries)', result['pool_reserved'] - result['pool_used'])
    except ImportError:
        pass
    print_fn('MEM summary')
    line('muGrid fields, all owners', total_fields)
    line('CuPy pool (reserved)', result['pool_reserved'])
    if result['process']:
        line('process on the device (nvidia-smi)', result['process'])
        line('= rest: CUDA context, FFT plans, ghost/MPI buffers', result['process'] - total_fields
             - result['pool_reserved'])
    return result
