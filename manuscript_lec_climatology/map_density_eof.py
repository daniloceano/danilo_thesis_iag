# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density_eof.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2025/01/31 14:40:32 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
from glob import glob

labels = ['A', 'B', 'C', 'D']

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                     color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

def plot_density(ax, density, eof, suffix, label):
    datacrs = ccrs.PlateCarree()
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat
    eof = int(eof)

    max_density = density.max().item()  # Obtém o valor máximo da densidade

    # Definir níveis de contorno
    if suffix == 'q90':
        if eof == 1:
            levels = [0.1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18]
        if eof in [2, 3]:
            levels = [0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9]
        elif eof == 4:
            levels = [0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

    if suffix == 'q10':
        if eof == 1:
            levels = [0.1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25]
        if eof in [2, 3, 4]:
            levels = [0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]
        # elif eof == 4:
        #     levels = [0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

    elif suffix == 'refined':
        if eof == 1:
            levels = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif eof == 2:
            levels = np.linspace(0.1, 1.4, 14)
        else:
            levels = np.linspace(0.1, 0.5, 11)

    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                     '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
               linewidths=0.35, linestyles='dashed', transform=datacrs)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(175, -25, f"({label}) EOF {eof}", ha='right', va='bottom', fontsize=14, fontweight='bold',
            bbox=props, zorder=101)

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
    gridlines(ax)

    return cf

def generate_density_panel(eofs_path, output_directory, suffix):
    eof_files = sorted(glob(os.path.join(eofs_path, "SAt_track_density_eof_*.nc")))

    os.makedirs(output_directory, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    for i, eof_file in enumerate(eof_files[:4]):  # Apenas as 4 primeiras EOFs
        eof_number = os.path.basename(eof_file).split('_')[-1].split('.')[0]
        ds = xr.open_dataset(eof_file)

        density = ds[f"EOF_{float(eof_number)}"]

        row, col = divmod(i, 2)  # Determina a posição no painel
        cf = plot_density(axes[row, col], density, eof=eof_number, suffix=suffix, label=labels[i])

        # Criar colorbar individual para cada subplot logo abaixo dele
        cbar_ax = fig.add_axes([0.12 + col * 0.47, 0.55 + row * -0.27, 0.3, 0.015])  # Posicionamento dinâmico
        cbar = plt.colorbar(cf, cax=cbar_ax, format='%g', orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95, hspace=-0.5, wspace=0.1)

    panel_path = os.path.join(output_directory, f"density_panel_{suffix}.png")
    plt.savefig(panel_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Density panel saved in {panel_path}')

if __name__ == "__main__":
    suffix = "q10"

    eofs_path = f"../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/csv_eofs_energetics_with_track/Total/track_density_{suffix}"
    output_directory = f"figures/eof_density_maps"

    generate_density_panel(eofs_path, output_directory, suffix)
