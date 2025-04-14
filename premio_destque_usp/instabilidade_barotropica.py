import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domínio
nx, ny = 30, 20
x = np.linspace(0, 10, nx)
y = np.linspace(0, 5, ny)
X, Y = np.meshgrid(x, y)

# Subamostragem para menos flechas
skip = (slice(None, None, 2), slice(None, None, 2))  # mostra 1 a cada 2

# Setup da figura
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_xticks([])
ax.set_yticks([])

# Inicialização do campo vetorial
U = np.zeros_like(X)
V = np.zeros_like(Y)
quiv = ax.quiver(X[skip], Y[skip], U[skip], V[skip], color='black', scale=10, width=0.01)

def update(frame):
    U = np.where(Y > 2.5, 1.0, -1.0)
    V = np.zeros_like(U)

    amp = 0.05 + 1.5 / (1 + np.exp(-0.1 * (frame - 35)))
    k = 2 * np.pi / 10

    for j in range(ny):
        for i in range(nx):
            dy = Y[j, i] - 2.5
            if abs(dy) < 1.5:
                U[j, i] += amp * np.cos(k * X[j, i]) * np.sin(np.pi * dy)
                V[j, i] += amp * np.sin(k * X[j, i]) * np.cos(np.pi * dy)

    quiv.set_UVC(U[skip], V[skip])
    return [quiv]

ani = FuncAnimation(fig, update, frames=90, interval=100, blit=True)
plt.tight_layout()
ani.save('instabilidade_barotropica.mp4', writer='ffmpeg', fps=20)
