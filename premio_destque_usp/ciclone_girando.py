import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Setup da figura
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-30, central_latitude=-30))
ax.set_global()
ax.coastlines(linewidth=0.6)
ax.add_feature(cfeature.LAND, facecolor='#a5d6a7')
ax.add_feature(cfeature.OCEAN, facecolor='#90e0ef')
ax.add_feature(cfeature.BORDERS, linewidth=0.2)

# Ciclone sobre o Atlântico Sul
cyclone_center_lon, cyclone_center_lat = -30, -38

n_arms = 4
n_points = 100
theta = np.linspace(0, 2*np.pi, n_points)
spirals = []

# Criar espirais com dados em coordenadas geográficas (lon/lat)
for i in range(n_arms):
    r = np.linspace(0.5, 2.5, n_points)
    offset = i * (2 * np.pi / n_arms)
    lon = cyclone_center_lon + r * np.cos(theta + offset)
    lat = cyclone_center_lat + r * np.sin(theta + offset)
    line, = ax.plot(lon, lat, color='white', lw=2, transform=ccrs.PlateCarree())
    spirals.append(line)

# Função de animação
def update(frame):
    for i, line in enumerate(spirals):
        r = np.linspace(5, 15, n_points)
        offset = i * (2 * np.pi / n_arms)
        phase = -0.1 * frame  # negativo = giro horário (SH)
        lon = cyclone_center_lon + r * np.cos(theta + offset + phase)
        lat = cyclone_center_lat + r * np.sin(theta + offset + phase)
        line.set_data(lon, lat)
    return spirals

ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.tight_layout()

# Salvar a animação
ani.save('ciclone_girando.mp4', writer='ffmpeg', fps=20)
