import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# 1. Carrega os dados
file = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered.csv'
df = pd.read_csv(file, parse_dates=['date'])

# Renomeia colunas para facilitar
df = df.rename(columns={'lon vor': 'lon', 'lat vor': 'lat'})
df = df.sort_values(by='date')

# Normaliza a intensidade (usaremos vor42 como intensidade)
df['intensity'] = df['vor42']
int_min, int_max = df['intensity'].min(), df['intensity'].max()

# Normaliza tamanho (opcional: escala visual)
df['size'] = 100 + 20 * (df['intensity'] - int_min) / (int_max - int_min)

# Lista de tempos únicos com aceleração
times = df['date'].drop_duplicates().sort_values().to_numpy()
step = 15
frames = list(range(0, len(times), step))

# Prepara cores
cmap = plt.cm.turbo
norm = plt.Normalize(int_min, int_max)

# 2. Setup do mapa
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, 30, -60, -10], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)

# Texto da data (no canto superior direito do mapa)
date_text = ax.text(25, -12, '', fontsize=14, weight='bold', transform=ccrs.PlateCarree(),
                    ha='right', va='top', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))

# 3. Inicializa os elementos
scat = ax.scatter([], [], s=[], c=[], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
trails = {}        # rastro acumulado por ciclone
trail_lines = {}   # linhas desenhadas para cada rastro

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

# 4. Atualiza a cada frame
def update(frame_idx):
    current_time = times[frame_idx]
    df_frame = df[df['date'] == current_time]
    print(f"Frame {frame_idx} | Data: {current_time} | Pontos: {len(df_frame)}")

    # Atualiza os rastros
    for _, row in df_frame.iterrows():
        tid = row['track_id']
        if tid not in trails:
            trails[tid] = []
        trails[tid].append((row['lon'], row['lat']))

    # Atualiza os pontos
    coords = df_frame[['lon', 'lat']].values
    sizes = df_frame['size'].values * 5
    colors = df_frame['intensity'].values

    scat.set_offsets(coords)
    scat.set_sizes(sizes)
    scat.set_array(colors)

    # Atualiza/desenha as linhas de rastro
    for tid, trail in trails.items():
        if tid in trail_lines:
            trail_lines[tid].set_data(*zip(*trail))
        else:
            line, = ax.plot(*zip(*trail), linewidth=2, color='gray', alpha=0.4, transform=ccrs.PlateCarree())
            trail_lines[tid] = line

    # Atualiza a data exibida
    date_formated = pd.to_datetime(current_time).strftime('%Y-%m-%d')
    date_text.set_text(date_formated)

    return [scat, date_text] + list(trail_lines.values())

# 5. Cria a animação
ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    blit=False, interval=50, repeat=False)

#plt.title('Ciclones extratropicais: intensidade, trajetória e data')
cbar = plt.colorbar(scat, ax=ax, label='', orientation='horizontal',
                    pad=0.08, aspect=20, shrink=0.3, extend='both')

# Remove os ticks e valores
cbar.set_ticks([])

# Adiciona apenas os extremos como texto manualmente
cbar.ax.text(-0.1, -0.8, 'menos intenso', transform=cbar.ax.transAxes,
             fontsize=10, ha='left', va='top')
cbar.ax.text(1.1, -0.8, 'mais intenso', transform=cbar.ax.transAxes,
             fontsize=10, ha='right', va='top')

plt.show()
