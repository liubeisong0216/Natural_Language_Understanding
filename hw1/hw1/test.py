import numpy as np
import matplotlib.pyplot as plt

# Parameter range
t_vals = np.linspace(0, 2*np.pi, 400)

# Parametric definitions
x_vals = t_vals * np.sin(t_vals)
y_vals = np.cos(t_vals)

# Plot
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, 'b-')
plt.plot([x_vals[0]], [y_vals[0]], 'ro', label='Start (t=0)')
plt.plot([x_vals[-1]], [y_vals[-1]], 'go', label='End (t=2π)')

# Axes & aspect
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.gca().set_aspect('equal', 'box')
plt.title(r"$x(t)=t\sin t,\;y(t)=\cos t,\;t\in[0,2\pi]$")
plt.legend()
plt.show()