import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Parameters
L = 0.001   # m (300 mm)
EI = 1.0 *1e-8    # Nm²
q0 = -1.005    # N/m

# ODE System: EI*w'''' = q0
def beam_ode(x, y):
    """y = [w, w', w'', w''']"""
    return np.vstack([y[1], y[2], y[3], q0/EI * np.ones_like(x)])

# Boundary Conditions - Simply Supported
def bc_simply_supported(ya, yb):
    return np.array([ya[0], ya[2], yb[0], yb[2]])  # w(0)=0, M(0)=0, w(L)=0, M(L)=0

# Boundary Conditions - Cantilever
def bc_cantilever(ya, yb):
    return np.array([ya[0], ya[1], yb[2], yb[3]])  # w(0)=0, w'(0)=0, M(L)=0, V(L)=0

# Solve
x_init = np.linspace(0, L, 20)
y_init = np.zeros((4, x_init.size))

sol_ss = solve_bvp(beam_ode, bc_simply_supported, x_init, y_init)
sol_cant = solve_bvp(beam_ode, bc_cantilever, x_init, y_init)

# Analytical Solutions
x = np.linspace(0, L, 200)
w_ana_ss = (q0/(24*EI)) * (x**4 - 2*L*x**3 + L**3*x)
w_ana_cant = (q0/(24*EI)) * (x**4 - 4*L*x**3 + 6*L**2*x**2)

# Numerical Solutions
w_num_ss = sol_ss.sol(x)[0]
w_num_cant = sol_cant.sol(x)[0]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Simply Supported - Deflection
axes[0,0].plot(x*1000, w_ana_ss*1000, 'b-', lw=2, label='Analytical')
axes[0,0].plot(x*1000, w_num_ss*1000, 'r--', lw=1.5, label='Numerical')
axes[0,0].set_title('Simply Supported Beam')
axes[0,0].set_xlabel('x [mm]')
axes[0,0].set_ylabel('Deflection [mm]')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Simply Supported - Error
error_ss = (w_num_ss - w_ana_ss) * 1e6
axes[1,0].plot(x*1000, error_ss, 'g-', lw=2)
axes[1,0].set_title('Error: Simply Supported')
axes[1,0].set_xlabel('x [mm]')
axes[1,0].set_ylabel('Error [μm]')
axes[1,0].axhline(0, color='k', ls=':', lw=0.8)
axes[1,0].grid(True, alpha=0.3)

# Cantilever - Deflection
axes[0,1].plot(x*1000, w_ana_cant*1000, 'b-', lw=2, label='Analytical')
axes[0,1].plot(x*1000, w_num_cant*1000, 'r--', lw=1.5, label='Numerical')
axes[0,1].set_title('Cantilever Beam')
axes[0,1].set_xlabel('x [mm]')
axes[0,1].set_ylabel('Deflection [mm]')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Cantilever - Error
error_cant = (w_num_cant - w_ana_cant) * 1e6
axes[1,1].plot(x*1000, error_cant, 'g-', lw=2)
axes[1,1].set_title('Error: Cantilever')
axes[1,1].set_xlabel('x [mm]')
axes[1,1].set_ylabel('Error [μm]')
axes[1,1].axhline(0, color='k', ls=':', lw=0.8)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Results
print("RESULTS")
print("-" * 50)
print(f"Parameters: L={L*1000}mm, EI={EI}Nm², q0={q0}N/m")
print("\nSimply Supported:")
print(f"  Max deflection (analytical): {np.min(w_ana_ss)*1000:.4f} mm")
print(f"  Max deflection (numerical):  {np.min(w_num_ss)*1000:.4f} mm")
print(f"  Max error: {np.max(np.abs(error_ss)):.3e} μm")
print("\nCantilever:")
print(f"  Tip deflection (analytical): {w_ana_cant[-1]*1000:.4f} mm")
print(f"  Tip deflection (numerical):  {w_num_cant[-1]*1000:.4f} mm")
print(f"  Max error: {np.max(np.abs(error_cant)):.3e} μm")