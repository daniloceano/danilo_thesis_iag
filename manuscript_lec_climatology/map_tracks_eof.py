import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines
import os

# Definir tamanhos para os rótulos
ylabel_fontsize = 18
title_fontsize = 20
tick_labelsize = 16
legend_fontsize = 14

# Definir caminhos
PATH = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/"
suffix = "refined"

eofs_path = f"{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_{suffix}.csv"
tracks_path = f"{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv"
output_directory = f"figures/eof_maps_tracks_{suffix}"

os.makedirs(output_directory, exist_ok=True)

# Carregar dados
tracks = pd.read_csv(tracks_path)
eofs = pd.read_csv(eofs_path)

# Fundir os dados pelo track_id para adicionar dominant_eof e região de gênese
merged_tracks = tracks.merge(eofs[['track_id', 'dominant_eof']], on='track_id')

# Manter apenas EOFs de 1 a 4
merged_tracks = merged_tracks[merged_tracks["dominant_eof"].isin([1, 2, 3, 4])]

# **Definir cores para as regiões de gênese**
region_colors = {
    "ARG": "#5975A4",      # Azul
    "LA-PLATA": "#CC8963", # Laranja
    "SE-BR": "#5F9E6E"     # Verde
}

# Labels para cada subplot
subplot_labels = ["(A)", "(B)", "(C)", "(D)"]

# Criar um painel de 2x2 para os mapas das EOFs
fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Iterar sobre as EOFs e criar um mapa para cada uma
for idx, (ax, eof) in enumerate(zip(axes.flat, sorted(merged_tracks["dominant_eof"].unique()))):
    subset = merged_tracks[merged_tracks["dominant_eof"] == eof]

    # Definir a extensão da região
    ax.set_extent([-90, 20, -60, 0], crs=ccrs.PlateCarree())

    # Adicionar elementos geográficos
    ax.add_feature(cfeature.LAND, color="lightgray")
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle="dotted", alpha=0.5)
    
    # Mostrar labels do grid apenas na esquerda e na parte de baixo
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {"size": tick_labelsize}
    gl.ylabel_style = {"size": tick_labelsize}

    legend_handles = [
        mlines.Line2D([], [], color=region_colors["ARG"], marker='o', linestyle='None', markersize=10, label="ARG"),
        mlines.Line2D([], [], color=region_colors["LA-PLATA"], marker='o', linestyle='None', markersize=10, label="LA-PLATA"),
        mlines.Line2D([], [], color=region_colors["SE-BR"], marker='o', linestyle='None', markersize=10, label="SE-BR")
    ]

    # Plotar as tracks de cada EOF, diferenciando por região de gênese
    for region, color in region_colors.items():
        region_subset = subset[subset["region"] == region]
        for track_id in region_subset["track_id"].unique():
            track = region_subset[region_subset["track_id"] == track_id]
            ax.plot(track["lon vor"], track["lat vor"],
                    color=color, linewidth=1, alpha=0.7,
                    marker='o', markersize=1,
                    transform=ccrs.PlateCarree())


    # Adicionar título com label em negrito
    ax.set_title(f"{subplot_labels[idx]} EOF {eof}", fontsize=title_fontsize, fontweight="bold")

    # Adicionar legenda apenas no primeiro subplot com marcadores maiores
    if idx == 0:
        legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=12, frameon=True)

    else:
        ax.legend().set_visible(False)

# Ajustar layout e salvar a figura
plt.tight_layout()
output_filepath = f"{output_directory}/eof_tracks_panel_by_region.png"
plt.savefig(output_filepath, dpi=300)
print(f"Painel salvo em: {output_filepath}")
