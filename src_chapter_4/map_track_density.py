# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_track_density.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/13 09:22:31 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/13 11:06:58 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os
import matplotlib as mpl

# Configuration
PHASES = ['incipient', 'intensification', 'mature', 'decay',
          'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = [False, "ARG", "LA-PLATA", "SE-BR"]
SEASONS = ['DJF', 'JJA']
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
COLORS = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
INFILES_DIRECTORY = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density/'
datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))


# Utility Functions
def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.8, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}


def plot_density(fig, ax, phase, density, label):
    # ax.set_extent([-90, 180, 0, -90], crs=datacrs)
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)

    levels_dict = {
        'incipient': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 30, 35],
        'intensification': [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130],
        'decay': [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130],
        'mature': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40],
        'default': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 28, 30]
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
        fig = plt.figure(figsize=(12, 5))
        for i, phase in enumerate(secondary_phases):
            ax = fig.add_subplot(2, 2, i+1, projection=proj)

            season_str = f"_{season}" if season else ""
            infile = os.path.join(INFILES_DIRECTORY, f'SAt_track_density{season_str}.nc')
            
            if not os.path.exists(infile):
                print(f"File not found: {infile}")
                continue

            ds = xr.open_dataset(infile)
            density = ds[phase]
            label = LABELS[i]
            levels, cf = plot_density(fig, ax, phase, density, label)

            if i == 3:
                cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
                ticks = np.round(levels, decimals=2)
                colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
                colorbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(wspace=0.15, hspace=-0.15)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_secondary_development{season_str}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Density map saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    plot_each_phase()
    plot_secondary_development()
