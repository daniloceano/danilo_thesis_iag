# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_track_density.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/13 09:22:31 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/15 10:32:45 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
import matplotlib as mpl
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Configuration
INFILES_DIRECTORY = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
# INFILES_DIRECTORY = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density/'
PHASES = ['incipient', 'intensification', 'mature', 'decay',
          'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = [False, "ARG", "LA-PLATA", "SE-BR"]
SEASONS = ['DJF', 'JJA']
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
COLORS = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', '#FEEC9F', '#FDB567', '#F06744', '#C1274A']

datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))

# Utility Functions
def gridlines(ax):
    # Create the gridlines with a finer interval
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed',
                      alpha=0.6, color='#383838', linewidth=0.5)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    
    # Use formatters to show labels
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # Control which labels are displayed
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False

    # Coarser interval
    gl2 = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0, color='#383838')
    gl2.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl2.ylocator = mticker.FixedLocator(np.arange(-90, 91, 20))
    
    # Use formatters to show labels
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()
    
    # Control which labels are displayed
    gl2.xlabel_style = {'size': 14, 'color': '#383838'}
    gl2.ylabel_style = {'size': 14, 'color': '#383838'}
    gl2.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.bottom_labels = False
    gl2.right_labels = False
    gl2.top_labels = True
    gl2.left_labels = True

def plot_density(fig, ax, phase, density, label):
    # ax.set_extent([-90, 180, 0, -90], crs=datacrs)
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)

    levels_dict = {
        'incipient': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 30, 35],
        'intensification': [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100],
        'mature': [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        'decay': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40],
        'default': [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7],
    }
    levels = levels_dict.get(phase, levels_dict['default'])

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = density.lon, density.lat

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838', linewidths=0.35, linestyles='dashed', transform=datacrs)
    ax.text(0.85, 0.85, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'), transform=ax.transAxes)
    ax.coastlines()
    gridlines(ax)
    return levels, cf

def load_and_sum_densities(phases, season):
    combined_density = None
    for region in ["ARG", "LA-PLATA", "SE-BR"]:
        region_str = region
        season_str = f"_{season}" if season else ""
        infile = os.path.join(INFILES_DIRECTORY, f'{region_str}_track_density{season_str}.nc')
        
        if os.path.exists(infile):
            ds = xr.open_dataset(infile)
            if combined_density is None:
                combined_density = ds[phases].copy()
            else:
                combined_density += ds[phases]
        else:
            print(f"File not found: {infile}")
    
    return combined_density

def plot_each_phase():
    for phase in PHASES:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 15), subplot_kw={'projection': proj})
        axes = axes.flatten()
        for i, region in enumerate(REGIONS):
            for j, season in enumerate(SEASONS):
                ax = axes[i*2 + j]

                if not region:
                    density = load_and_sum_densities(phase, season)
                    region_str = "SAt"
                else:
                    region_str = region
                    season_str = f"_{season}" if season else ""
                    infile = os.path.join(INFILES_DIRECTORY, f'{region_str}_track_density{season_str}.nc')
                    
                    if not os.path.exists(infile):
                        print(f"File not found: {infile}")
                        continue

                    ds = xr.open_dataset(infile)
                    density = ds[phase]

                label = LABELS[i*2 + j]
                levels, cf = plot_density(fig, ax, phase, density, label)

                if i*2 + j == 7:
                    cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
                    ticks = np.round(levels, decimals=2)
                    colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
                    colorbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(wspace=0.15)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_{phase}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Density map saved in {fname}')

def plot_secondary_development():
    secondary_phases = ['intensification 2', 'mature 2', 'decay 2', 'residual']
    for season in SEASONS:
        fig = plt.figure(figsize=(12, 7))
        for i, phase in enumerate(secondary_phases):
            ax = fig.add_subplot(2, 2, i+1, projection=proj)

            density = load_and_sum_densities(phase, season)
            if density is None:
                print(f"No data found for phase: {phase} in season: {season}")
                continue

            label = LABELS[i]
            levels, cf = plot_density(fig, ax, phase, density, label)

            if i == 3:
                cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
                ticks = np.round(levels, decimals=2)
                colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
                colorbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(wspace=0.15)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_secondary_development_{season}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Density map saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    plot_each_phase()
    plot_secondary_development()
