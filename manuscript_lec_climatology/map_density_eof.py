# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density_eof.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2025/01/30 15:49:12 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Script to plot cyclone density for each EOF.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
from glob import glob

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                     color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

def plot_density(ax, fig, density, eof, suffix):
    datacrs = ccrs.PlateCarree()
    
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat
    eof = int(eof)

    if suffix == 'q90':
        if eof == 1:
            levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
        if eof in [2, 3]:
            levels = [0.1, 1, 2, 5, 8, 10, 13, 15, 18, 20, 25, 30]
        elif eof == 4:
            levels = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif eof == 5:
            levels = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        elif eof > 5:
            levels = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

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

    cbar_axes = fig.add_axes([0.15, 0.26, 0.7, 0.02])
    ticks = np.round(levels, decimals=2)
    colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
    colorbar.ax.tick_params(labelsize=12)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(175, -25, f"({labels[eof - 1]}) EOF {eof}", ha='right', va='bottom', fontsize=14, fontweight='bold',
            bbox=props, zorder=101)

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
    gridlines(ax)

def generate_density_map_for_eofs(eofs_path, output_directory, suffix):
    eof_files = sorted(glob(os.path.join(eofs_path, "SAt_track_density_eof_*.nc")))

    os.makedirs(output_directory, exist_ok=True)

    for eof_file in eof_files:
        eof_number = os.path.basename(eof_file).split('_')[-1].split('.')[0]
        eof_number = float(eof_number)
        ds = xr.open_dataset(eof_file)
        density = ds[f"EOF_{eof_number}"]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        plot_density(ax, fig, density, eof=eof_number, suffix=suffix)

        fname = os.path.join(output_directory, f"density_eof_{eof_number}.png")
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f'Density map for EOF {eof_number} saved in {fname}')

if __name__ == "__main__":
    # Suffix options: "refined" or "q90"
    suffix = "refined"

    eofs_path = f"../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/csv_eofs_energetics_with_track/Total/track_density_{suffix}"
    output_directory = f"figures/eof_density_maps_{suffix}"

    generate_density_map_for_eofs(eofs_path, output_directory, suffix)
