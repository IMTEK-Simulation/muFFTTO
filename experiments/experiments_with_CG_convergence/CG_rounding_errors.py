import numpy as np
import matplotlib.pyplot as plt
def cg(A, b, x0=None, maxiter=None):
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    history = [np.linalg.norm(r)]

    for k in range(maxiter or n):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        history.append(np.linalg.norm(r))

        if np.sqrt(rs_new) < 1e-14:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, np.array(history)

# Construct ill-conditioned SPD matrix
n = 50
eigs = np.logspace(0, -3, n)
A = np.diag(eigs)
b = np.ones(n)

x, hist = cg(A, b, maxiter=2000)
plt.semilogy( hist)
plt.title(f'{{eigs = np.logspace(0, -3, n)}}')
plt.show()
print("Error:", 1/x-eigs)
print("Final residual:", hist[-1])
