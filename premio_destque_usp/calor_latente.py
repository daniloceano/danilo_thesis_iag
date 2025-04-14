import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from matplotlib.path import Path

# Função para criar setas curvas (com direção configurável)
def curved_heat_arrow(x0, y0, dx=1.0, dy=1.5, side='right', upward=True, color='#d62828'):
    if side == 'right':
        x_sign = 1
    else:
        x_sign = -1

    y_sign = 1 if upward else -1

    # Curva em S (2 inflexões usando pontos de controle simétricos e invertidos)
    verts = [
        (x0, y0),  # Início
        (x0 + x_sign * dx * 0.5, y0 + y_sign * dy * 0.2),
        (x0 - x_sign * dx * 0.5, y0 + y_sign * dy * 0.6),
        (x0 + x_sign * dx, y0 + y_sign * dy),  # Fim
    ]

    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)

    arrow = FancyArrowPatch(
        path=path,
        arrowstyle='-|>',
        mutation_scale=20,
        lw=2,
        color=color,
        visible=False
    )
    return arrow

# Setup da figura
fig, ax = plt.subplots(figsize=(5, 7))
ax.set_xlim(-6, 6)
ax.set_ylim(0, 12)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Cumulonimbus com Liberação de Calor Latente em Todas as Direções")

# Solo
ax.axhline(y=0.5, color='forestgreen', linewidth=6)

# Camadas da nuvem
cloud_layers = [
    {"patch": Circle((0, 2.5), 0.9, color='#E9ECEF', visible=False), "trigger": 2.0},
    {"patch": Circle((0, 4.2), 1.1, color='#DEE2E6', visible=False), "trigger": 4.0},
    {"patch": Circle((0, 6.5), 1.3, color='#CED4DA', visible=False), "trigger": 6.2},
]

# Criando as setas para cada camada
layer_arrows = []
for center in [2.5, 4.2, 6.5]:
    # Direita - para cima e para baixo
    layer_arrows.append(curved_heat_arrow(1.2, center + 0.1, dx=0.6, dy=0.7, side='right', upward=True))
    layer_arrows.append(curved_heat_arrow(1.2, center - 0.1, dx=0.6, dy=0.7, side='right', upward=False))
    # Esquerda - para cima e para baixo
    layer_arrows.append(curved_heat_arrow(-1.2, center + 0.1, dx=0.6, dy=0.7, side='left', upward=True))
    layer_arrows.append(curved_heat_arrow(-1.2, center - 0.1, dx=0.6, dy=0.7, side='left', upward=False))

# Patch das nuvens
for layer in cloud_layers:
    ax.add_patch(layer["patch"])
for arr in layer_arrows:
    ax.add_patch(arr)

# Anvil (bigorna)
anvil = Ellipse((0, 8), width=4.5, height=1.0, color='#ADB5BD', visible=False)
ax.add_patch(anvil)

# Setas do topo (3 curvadas: esquerda/cima, reta cima, direita/cima)
top_arrows = [
    curved_heat_arrow(-0.8, 8.5, dx=0.7, dy=1.0, side='left', upward=True),
    curved_heat_arrow(0, 8.5, dx=0.1, dy=1.0, side='right', upward=True),  # praticamente reta
    curved_heat_arrow(0.8, 8.5, dx=0.7, dy=1.0, side='right', upward=True),
]
for ta in top_arrows:
    ax.add_patch(ta)

# Seta principal de ascensão
ascent_arrow = FancyArrowPatch(posA=(0, 0.6), posB=(0, 0.7), color='#bc4749',
                                arrowstyle='-|>', mutation_scale=20, zorder=100)
ax.add_patch(ascent_arrow)

# Texto: convecção (na base da seta)
text_convection = ax.text(0.2, 1.0, "convecção", fontsize=12, color='#bc4749', visible=False)

# Texto: formação de nuvem (ao lado da primeira camada)
text_cloud = ax.text(1.5, 2.5, "formação de nuvem", fontsize=12, color='gray', visible=False)

# Texto: liberação de calor latente (próximo da primeira seta curva)
text_latent_heat = ax.text(2.2, 2.8, "liberação de\n calor latente", fontsize=11, color='#d62828', visible=False)

def update(frame):
    max_height = 8  # altura máxima da pluma (topo da nuvem)
    height = min(max_height, 0.1 * frame)
    ascent_arrow.set_positions((0, 0.6), (0, 0.6 + height))

    if frame >= 1:
        text_convection.set_visible(True)

    for i, layer in enumerate(cloud_layers):
        if height >= layer["trigger"]:
            layer["patch"].set_visible(True)
            for j in range(4):
                layer_arrows[i*4 + j].set_visible(True)

            if i == 0:
                text_cloud.set_visible(True)
                text_latent_heat.set_visible(True)

    if height >= 7.8:
        anvil.set_visible(True)

    if frame >= 80:
        for ta in top_arrows:
            ta.set_visible(True)

    return [ascent_arrow, text_convection, text_cloud, text_latent_heat] + [l["patch"] for l in cloud_layers] + layer_arrows + [anvil] + top_arrows

ani = FuncAnimation(fig, update, frames=90, interval=100, blit=True)

plt.tight_layout()

# Salvar a animação
ani.save('calor_latente.mp4', writer='ffmpeg', fps=20)