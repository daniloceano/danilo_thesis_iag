import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import numpy as np

# Setup da figura
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Pontos dos vértices
baroclinica = (8.5, 8.5)
calor_latente = (1.5, 8.5)
barotropica = (5, 2)

# Rótulos em português
ax.text(*calor_latente, "Liberação de\ncalor latente", ha='center', va='center', fontsize=20, fontweight='bold', color='#bc4749')
ax.text(*baroclinica, "Instabilidade\nbaroclínica", ha='center', va='center', fontsize=20, fontweight='bold', color='#5a189a')
ax.text(*barotropica, "Instabilidade\nbarotrópica", ha='center', va='center', fontsize=20, fontweight='bold', color='#386641')

# Triângulo das instabilidades
ax.plot([calor_latente[0], barotropica[0]], [calor_latente[1], barotropica[1]], 'k--', alpha=0.3)
ax.plot([baroclinica[0], barotropica[0]], [baroclinica[1], barotropica[1]], 'k--', alpha=0.3)
ax.plot([calor_latente[0], baroclinica[0]], [calor_latente[1], baroclinica[1]], 'k--', alpha=0.3)

# Mancha inicial (grande e horizontal no topo)
start_center = ((calor_latente[0] + baroclinica[0]) / 2, 8.5)

# Posição final: mais próxima da barotrópica, mas ainda puxando para cima
end_center = (5, 5.7)

# Mancha (ciclones extratropicais)
shade = Ellipse(xy=start_center, width=7.5, height=1.0, color='skyblue', alpha=0.8)
ax.add_patch(shade)

# Texto dentro da mancha (fixado ao centro do shade)
shade_text = ax.text(
    *start_center,
    "Ciclones\nextratropicais",
    ha='center',
    va='center',
    fontsize=20,
    color='black',
    weight='bold'
)

# Título
title = ax.text(5, 9.5, '', ha='center', va='center', fontsize=30, weight='bold')

# Função de animação
def update(frame):
    t = frame / 60

    new_x = (1 - t) * start_center[0] + t * end_center[0]
    new_y = (1 - t) * start_center[1] + t * end_center[1]
    shade.center = (new_x, new_y)
    shade_text.set_position((new_x, new_y))

    new_width = 7.5 - 3.0 * t
    new_height = 1.0 + 2.5 * t
    shade.width = new_width
    shade.height = new_height

    if frame < 30:
        title.set_text("Conhecimento prévio")
    else:
        title.set_text("Conclusão da minha tese")

    return [shade, shade_text, title]

ani = FuncAnimation(fig, update, frames=60, interval=100, blit=True)

plt.tight_layout()

# Salvar a animação
ani.save('instabilidade_conclusao.mp4', writer='ffmpeg', fps=20)