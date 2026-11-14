import muGrid
print(getattr(muGrid, "__version__", ""), [n for n in ("GridTransfer", "linalg", "memory_profile", "use_cupy_allocator", "route_cupy_through_mugrid", "IsotropicStiffnessOperator3D", "CartesianDecomposition") if hasattr(muGrid, n)])
print([n for n in dir(muGrid.linalg) if not n.startswith("_")])
print(muGrid.has_mpi, muGrid.has_cuda)
