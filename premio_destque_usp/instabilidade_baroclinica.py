import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domínio espacial
nx, ny = 200, 100
x = np.linspace(0, 10, nx)
y = np.linspace(0, 5, ny)
X, Y = np.meshgrid(x, y)

# Setup da figura
fig, ax = plt.subplots()
cmap = plt.get_cmap("coolwarm")
T = np.zeros_like(X)
im = ax.imshow(T, origin='lower', extent=[0, 5, 0, 5], cmap=cmap, vmin=0, vmax=40)
# ax.set_title("Instabilidade Baroclínica em Desenvolvimento")

# Remover ticks dos eixos
ax.set_xticks([])
ax.set_yticks([])

# # Anotar "ar quente" e "ar frio"
# ax.annotate("Ar Quente", xy=(3, 2), xytext=(2, 3),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             fontsize=12, color='red')
# ax.annotate("Ar Frio", xy=(8, 3), xytext=(7, 4),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             fontsize=12, color='blue')

def temperature_field(frame):
    T = np.zeros_like(X)
    
    # Crescimento suave da amplitude com sigmoide (fase inicial quase estável, depois acelera)
    t = frame
    amp = 0.05 + 1.5 / (1 + np.exp(-0.1 * (t - 60)))  # sigmoid centered at frame 35

    # Frente com ondulação crescente
    front_y = 2.5 + amp * np.sin(2 * np.pi * X[0, :] / 10)

    for j in range(ny):
        for i in range(nx):
            T[j, i] = 10 if Y[j, i] > front_y[i] else 30

    return T

def update(frame):
    T = temperature_field(frame)
    im.set_data(T)
    return [im]

ani = FuncAnimation(fig, update, frames=90, interval=20, blit=True)
plt.tight_layout()

# plt.show()

# # Salvar a animação
# ani.save('instabilidade_baroclinica.gif', writer='imagemagick', fps=10)
ani.save('instabilidade_baroclinica.mp4', writer='ffmpeg', fps=20)
# plt.close(fig)  # Fecha a figura após salvar