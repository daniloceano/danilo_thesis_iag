import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Configuration
INFILES_DIRECTORY = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../animations/'
PHASES = ['incipient', 'intensification', 'mature', 'decay',
          'intensification 2', 'mature 2', 'decay 2']
REGIONS = ["ARG", "LA-PLATA", "SE-BR"]
SEASONS = ['DJF', 'JJA']
COLORS = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', '#FEEC9F', '#FDB567', '#F06744', '#C1274A']

datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Utility Functions
def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed',
                      alpha=0.6, color='#383838', linewidth=0.5)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False

    gl2 = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0, color='#383838')
    gl2.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl2.ylocator = mticker.FixedLocator(np.arange(-90, 91, 20))
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()
    gl2.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.bottom_labels = False
    gl2.right_labels = False
    gl2.top_labels = True
    gl2.left_labels = True

def plot_density(ax, phase, density, label):
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)
    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)

    levels_dict = {
        'incipient': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 30, 35],
        'intensification': [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100],
        'mature': [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        'decay': [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40],
        'intensification 2': [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
        'mature 2': [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
        'decay 2': [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7],
        'residual': [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    }
    
    levels = levels_dict.get(phase)

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = density.lon, density.lat

    cf = ax.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=datacrs)
    ax.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838', linewidths=0.35, linestyles='dashed', transform=datacrs)
    ax.text(0.85, 0.85, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white'), transform=ax.transAxes)
    ax.coastlines()
    gridlines(ax)
    return cf

def load_density(region, season, phase):
    season_str = f"_{season}" if season else ""
    infile = os.path.join(INFILES_DIRECTORY, f'{region}_track_density{season_str}.nc')
    
    if os.path.exists(infile):
        ds = xr.open_dataset(infile)
        return ds[phase]
    else:
        print(f"File not found: {infile}")
        return None

# Create the animation
fig, axes = plt.subplots(len(REGIONS), len(SEASONS), figsize=(12, 13), subplot_kw={'projection': proj})
axes = axes.flatten()

def update(frame):
    phase = PHASES[frame]
    fig.suptitle(f'Phase: {phase}', fontsize=22, y=0.95, fontweight='bold')
    for i, region in enumerate(REGIONS):
        for j, season in enumerate(SEASONS):
            ax = axes[i*len(SEASONS) + j]
            density = load_density(region, season, phase)
            if density is not None:
                ax.clear()
                cf = plot_density(ax, phase, density, f'')
                if i == 0:  # Add season label at the top
                    ax.set_title(season, fontsize=20, pad=30)
                if j == 0:  # Add region label on the left
                    ax.text(-0.2, 0.5, region, va='center', ha='center', rotation='vertical', rotation_mode='anchor', fontsize=20, transform=ax.transAxes)
    return axes

ani = animation.FuncAnimation(fig, update, frames=len(PHASES), repeat=True)

# Save the animation
ani.save(os.path.join(OUTPUT_DIRECTORY, 'cyclone_life_cycle_animation.mp4'), writer='ffmpeg', fps=1)

print(f'Animation saved in {os.path.join(OUTPUT_DIRECTORY, "cyclone_life_cycle_animation.mp4")}')