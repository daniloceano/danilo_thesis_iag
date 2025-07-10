import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
from glob import glob

labels = ['A', 'B', 'C', 'D', 'E']  # Agora apenas 5 labels

# Regiões de gênese
regions = {
    "SE-BR": [(-52, -38, -37, -23)],
    "LA-PLATA": [(-69, -38, -52, -23)],
    "ARG": [(-70, -55, -50, -39)],
}

def gridlines(ax):
    """Configura as linhas de grade nos mapas"""
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                     color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-90, 101, 30))  # Ajuste para 20° exatos
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838'}

def generate_density_panel(cluster_density_path, output_directory):
    """Gera um painel de densidade dos clusters com 3x2 subplots e remove espaço extra
    Adiciona um colorbar único para todas as figuras, usando o valor máximo global de densidade."""
    cluster_files = sorted(glob(os.path.join(cluster_density_path, "track_density_cluster_*.nc")))

    os.makedirs(output_directory, exist_ok=True)

    # Corrige erro de 'projection' para subplots quando axes é 2D
    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                            subplot_kw=dict(projection=ccrs.PlateCarree()))
    axes = np.array(axes)  # Garante que axes é um array numpy para indexação

    # Encontrar o valor máximo global de densidade
    densities = []
    for cluster_file in cluster_files:
        cluster_number = os.path.basename(cluster_file).split('_')[-1].split('.')[0]
        ds = xr.open_dataset(cluster_file)
        density = ds[f"Cluster {int(cluster_number)}"]
        densities.append(density)
        ds.close()

    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
            '#FEEC9F', '#FDB567', '#F06744', '#C1274A']


    for i, (cluster_file, density) in enumerate(zip(cluster_files, densities)):
        max_density = density.max().values
        levels = np.linspace(0.1, max_density, 9)
        levels = np.round(levels, 1)
        cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)

        cluster_number = int(os.path.basename(cluster_file).split('_')[-1].split('.')[0])
        row, col = divmod(i, 2)  # Determina a posição no painel
        ax = axes[row, col]
        ax.set_extent([-90, 110, -15, -90], crs=ccrs.PlateCarree())
        cf = ax.contourf(density.lon, density.lat, density, cmap=cmap, levels=levels, norm=norm, extend='max', transform=ccrs.PlateCarree())
        ax.contour(density.lon, density.lat, density, levels=levels, norm=norm, colors='#383838',
                   linewidths=0.35, linestyles='dashed', transform=ccrs.PlateCarree())
        props = dict(boxstyle='round', facecolor='white')
        ax.text(80, -30, f"({labels[i]}) cluster {cluster_number}", ha='right', va='bottom', fontsize=14, fontweight='bold',
                bbox=props, zorder=101)
        ax.coastlines(zorder=1)
        ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
        gridlines(ax)

        # Adicionar regiões de gênese
        for name, bounds in regions.items():
            min_lon, min_lat, max_lon, max_lat = bounds[0]
            ax.plot(
                [min_lon, max_lon, max_lon, min_lon, min_lon],
                [min_lat, min_lat, max_lat, max_lat, min_lat],
                color='black', linewidth=1.5, transform=ccrs.PlateCarree(), linestyle='--'
            )

        # Adicionar colorbar horizontal abaixo de cada subplot
        cbar = fig.colorbar(cf, orientation='horizontal', extend='max', pad=0.08)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(h_pad=-12)
    panel_path = os.path.join(output_directory, "density_panel.png")
    plt.savefig(panel_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Density panel saved in {panel_path}')

if __name__ == "__main__":
    cluster_density_path = "track_density_clusters"
    output_directory = "figures/eof_clusters_intense"
    generate_density_panel(cluster_density_path, output_directory)
