# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_track_density_difference.py                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/13 09:22:31 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/13 19:18:59 by daniloceano      ###   ########.fr        #
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
INFILES_DIRECTORY = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density_difference/'
PHASES = ['incipient', 'intensification', 'mature', 'decay',
          'intensification 2', 'mature 2', 'decay 2', 'residual']
SEASONS = ['DJF', 'JJA']
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
COLORS = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', '#FEEC9F', '#FDB567', '#F06744', '#C1274A']

datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))

# Utility Functions
def gridlines(ax):
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

def plot_density_difference(fig, ax, phase, density_diff, label, season):
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)
    
    levels = np.linspace(-1, 1, 21)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = density_diff.lon, density_diff.lat

    cf = ax.contourf(lon, lat, density_diff, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density_diff, levels=levels, norm=norm, colors='#383838', linewidths=0.35, linestyles='dashed', transform=datacrs)
    ax.text(0.85, 0.85, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'), transform=ax.transAxes)
    ax.coastlines()
    gridlines(ax)
    return levels, cf

def load_and_sum_densities(phase, season):
    combined_density = None
    for region in ["ARG", "LA-PLATA", "SE-BR"]:
        region_str = region
        season_str = f"_{season}" if season else ""
        infile = os.path.join(INFILES_DIRECTORY, f'{region_str}_track_density{season_str}.nc')
        
        if os.path.exists(infile):
            ds = xr.open_dataset(infile)
            if combined_density is None:
                combined_density = ds[phase].copy()
            else:
                combined_density += ds[phase]
        else:
            print(f"File not found: {infile}")
    
    return combined_density

def normalize_density(density):
    return (density - density.min()) / (density.max() - density.min())

def plot_phase_differences():
    phase_pairs = [
        ('incipient', 'intensification'),
        ('intensification', 'mature'),
        ('mature', 'decay'),
        ('decay', 'intensification 2'),
        ('intensification 2', 'mature 2'),
        ('mature 2', 'decay 2')
    ]  # Excluding residual phase
    for phase_prev, phase in phase_pairs:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), subplot_kw={'projection': proj})
        for season, ax, label in zip(SEASONS, axes, ['(A)', '(B)']):
            density_prev = normalize_density(load_and_sum_densities(phase_prev, season))
            density = normalize_density(load_and_sum_densities(phase, season))
            density_diff = density - density_prev

            levels, cf = plot_density_difference(fig, ax, phase, density_diff, label, season)

        cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
        ticks = np.linspace(-1, 1, 11)
        colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
        colorbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_difference_map_{phase_prev}_to_{phase}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Density difference map saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    plot_phase_differences()