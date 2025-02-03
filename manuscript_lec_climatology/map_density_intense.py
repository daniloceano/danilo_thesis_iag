import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

labels = ['A', 'B', 'C', 'D', 'E']  # Agora apenas 5 labels

def gridlines(ax):
    """Configura as linhas de grade nos mapas"""
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                     color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

def plot_density(ax, density, cluster, label):
    """Plota a densidade de tracks no mapa"""
    datacrs = ccrs.PlateCarree()
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat
    cluster = int(cluster)

    max_density = density.max().item()  # Obtém o valor máximo da densidade
    levels = np.linspace(0.1, max_density, 11)
    levels = np.round(levels, decimals=2)

    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                     '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
               linewidths=0.35, linestyles='dashed', transform=datacrs)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(170, -25, f"({label}) cluster {cluster}", ha='right', va='bottom', fontsize=14, fontweight='bold',
            bbox=props, zorder=101)

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
    gridlines(ax)

    return cf

def generate_density_panel(cluster_density_path, output_directory):
    """Gera um painel de densidade dos clusters com 3x2 subplots e remove espaço extra"""
    cluster_files = sorted(glob(os.path.join(cluster_density_path, "track_density_cluster_*.nc")))

    os.makedirs(output_directory, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), subplot_kw={"projection": ccrs.PlateCarree()})

    for i, cluster_file in enumerate(cluster_files):
        cluster_number = os.path.basename(cluster_file).split('_')[-1].split('.')[0]
        ds = xr.open_dataset(cluster_file)

        density = ds[f"Cluster {int(cluster_number)}"]

        row, col = divmod(i, 2)  # Determina a posição no painel
        cf = plot_density(axes[row, col], density, cluster=cluster_number, label=labels[i])

        # Criar colorbar diretamente vinculada à figura (evita erro do Cartopy)
        cbar = fig.colorbar(cf, ax=axes[row, col], orientation='horizontal', fraction=0.08, pad=0.1)
        cbar.ax.tick_params(labelsize=10)

        ds.close()  # Fechar o dataset após leitura

    # Remover subplot extra caso tenha menos de 6 clusters
    if len(cluster_files) < 6:
        fig.delaxes(axes[-1, -1])  # Remove o último subplot vazio

    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, hspace=-0.2, wspace=0.1)

    panel_path = os.path.join(output_directory, "density_panel.png")
    plt.savefig(panel_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Density panel saved in {panel_path}')

if __name__ == "__main__":

    cluster_density_path = "track_density_clusters"
    output_directory = "figures/eof_clusters_intense"

    generate_density_panel(cluster_density_path, output_directory)
