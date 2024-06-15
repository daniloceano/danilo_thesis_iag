# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_track_density_secondary.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/15 10:32:45 by danilo.oceano      #+#    #+#              #
#    Updated: 2024/06/15 13:45:00 by danilo.oceano      ###   ########.fr        #
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Configuration
INFILES_DIRECTORY = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
SECONDARY_INFILES_DIRECTORY = '../results_chapter_4/track_density_secondary_development/'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density/'
PHASES = ['intensification 2', 'mature 2', 'decay 2']
REGIONS = ["ARG", "LA-PLATA", "SE-BR"]
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
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
    gl2.xlocator = mticker.FixedLocator(np.arange(-180, 181, 40))
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

def plot_density(fig, ax, density, label):
    ax.set_extent([-60, 130, -15, -70], crs=datacrs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)

    levels = np.linspace(0.1, np.amax(density) + 0.1, 12)
    levels = np.round(levels, decimals=2)

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = density.lon, density.lat

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838', linewidths=0.35, linestyles='dashed', transform=datacrs)
    ax.text(0.85, 0.85, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'), transform=ax.transAxes)
    ax.coastlines()
    gridlines(ax)
    return levels, cf

def load_and_sum_densities(phases, directory):
    combined_density = None
    for region in REGIONS:
        infile = os.path.join(directory, f'{region}_track_density.nc')
        
        if os.path.exists(infile):
            ds = xr.open_dataset(infile)
            summed_density = ds[phases[0]].copy()
            for phase in phases[1:]:
                summed_density += ds[phase]
            if combined_density is None:
                combined_density = summed_density
            else:
                combined_density += summed_density
        else:
            print(f"File not found: {infile}")
    
    return combined_density

def plot_secondary_development(directory1, directory2, output_suffix):
    secondary_phases = ['intensification 2', 'decay 2']
    fig, axes = plt.subplots(nrows=len(secondary_phases), ncols=2, figsize=(12, 12), subplot_kw={'projection': proj})
    for i, phase in enumerate(secondary_phases):
        for j, (directory, label_suffix) in enumerate([(directory1, 'original'), (directory2, 'secondary')]):
            ax = axes[i, j]

            density = load_and_sum_densities([phase], directory)
            if density is None:
                print(f"No data found for phase: {phase} in directory: {label_suffix}")
                continue

            label = f"{LABELS[i * 2 + j]}"
            levels, cf = plot_density(fig, ax, density, label)

            cbar_axes = inset_axes(ax, width="70%", height="10%", loc='lower center',
                       bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            
            colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=levels, format='%g', orientation='horizontal')
            colorbar.ax.tick_params(labelsize=12, rotation=45)

    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_secondary_development_{output_suffix}.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Density map saved in {fname}')

def plot_secondary_individual_regions(directory, output_suffix):
    secondary_phases = ['intensification 2', 'mature 2', 'decay 2']
    for phase in secondary_phases:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), subplot_kw={'projection': proj})
        regions = ["ALL"] + REGIONS
        for i, region in enumerate(regions):
            ax = axes.flat[i]

            if region == "ALL":
                density = load_and_sum_densities([phase], directory)
            else:
                infile = os.path.join(directory, f'{region}_track_density.nc')
                if not os.path.exists(infile):
                    print(f"File not found: {infile}")
                    continue

                ds = xr.open_dataset(infile)
                density = ds[phase].copy()

            if density is None:
                print(f"No data found for region: {region}")
                continue

            label = f"({chr(65 + i)})"
            levels, cf = plot_density(fig, ax, density, label)

            cbar_axes = inset_axes(ax, width="70%", height="10%", loc='lower center',
                       bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

            colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=levels, format='%g', orientation='horizontal')
            colorbar.ax.tick_params(labelsize=12, rotation=45)

        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_{phase}_regions_{output_suffix}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Density map saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    plot_secondary_development(INFILES_DIRECTORY, SECONDARY_INFILES_DIRECTORY, 'combined')
    plot_secondary_individual_regions(SECONDARY_INFILES_DIRECTORY, 'secondary')