import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os

# Caminhos dos arquivos
arg_density_path = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density/ARG_track_density.nc'
la_plata_density_path = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density/LA-PLATA_track_density.nc'
se_br_density_path = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density/SE-BR_track_density.nc'

# Regiões de gênese
regions = {
    "SE-BR": [(-52, -38, -37, -23)],
    "LA-PLATA": [(-69, -38, -52, -23)],
    "ARG": [(-70, -55, -50, -39)],
}

# Coordenadas personalizadas para as etiquetas
region_labels = {
    "SE-BR": (-40, -19),  # Coordenadas ajustadas para SE-BR
    "LA-PLATA": (-65, -19),  # Coordenadas ajustadas para LA-PLATA
    "ARG": (-60, -62),  # Coordenadas ajustadas para ARG
}

# Função para configurar as linhas do grid
def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                      color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

# Função para adicionar as regiões de gênese ao mapa
def add_regions(ax):
    for name, bounds in regions.items():
        min_lon, min_lat, max_lon, max_lat = bounds[0]
        ax.plot(
            [min_lon, max_lon, max_lon, min_lon, min_lon],
            [min_lat, min_lat, max_lat, max_lat, min_lat],
            color='black', linewidth=1.5, transform=ccrs.PlateCarree(), linestyle='--'
        )
        # Adicionar a etiqueta usando as coordenadas personalizadas
        label_lon, label_lat = region_labels[name]
        ax.text(
            label_lon, label_lat, name, fontsize=12,
            color='black', fontweight='bold', ha='center', transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.7)
        )

# Função para plotar a densidade combinada
def plot_combined_density(ax, density):
    datacrs = ccrs.PlateCarree()
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat

    levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                     '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
               linewidths=0.35, linestyles='dashed', transform=datacrs)

    cbar_axes = plt.gcf().add_axes([0.15, 0.26, 0.7, 0.02])
    ticks = np.round(levels, decimals=2)
    plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
    gridlines(ax)
    add_regions(ax)  # Adicionar as regiões de gênese

# Carregar os dados de densidade
arg_density = xr.open_dataset(arg_density_path)
la_plata_density = xr.open_dataset(la_plata_density_path)
se_br_density = xr.open_dataset(se_br_density_path)

# Somar as densidades para todas as fases
phases = ['intensification', 'incipient', 'mature', 'decay', 
          'residual', 'intensification 2', 'mature 2', 'decay 2']

arg_combined = sum(arg_density[phase] for phase in phases)
la_plata_combined = sum(la_plata_density[phase] for phase in phases)
se_br_combined = sum(se_br_density[phase] for phase in phases)

# Somar as densidades para todas as regiões
combined_density = arg_combined + la_plata_combined + se_br_combined

# Plotar a densidade combinada
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
plot_combined_density(ax, combined_density)

# Salvar a figura
output_path = 'figures/combined_density_map_all_phases_regions_with_regions.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
print(f'Combined density map saved to {output_path}')
