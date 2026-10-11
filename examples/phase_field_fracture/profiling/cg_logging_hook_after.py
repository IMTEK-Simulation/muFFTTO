
# ---- save convergence logs (test hook) ----
if rank == 0 and output_folder is not None:
    import pickle
    with open(os.path.join(output_folder, 'cg_convergence.pkl'), 'wb') as f:
        pickle.dump({'steihaug': cg_log, 'diagnostic': diag_log,
                     'mechanics_preconditioner': mechanics_preconditioner}, f)
    root_print(f'wrote {os.path.join(output_folder, "cg_convergence.pkl")}')
