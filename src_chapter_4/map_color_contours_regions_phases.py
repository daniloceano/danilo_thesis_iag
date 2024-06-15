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
INFILES_DIRECTORY = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density_regions/'
PHASES = ['incipient', 'intensification', 'mature', 'decay',
          'intensification 2', 'mature 2', 'decay 2']
REGIONS = ["ARG", "LA-PLATA", "SE-BR"]
SEASONS = ['DJF', 'JJA']
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']

COLOR_PHASES = {
    'Total': '#1d3557',
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'intensification 2': '#ca6702',
    'mature': '#d62828',
    'mature 2': '#9b2226',
    'decay': '#9aa981',
    'decay 2': '#386641',
}

datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed',
                      alpha=0.6, color='#383838', linewidth=0.5)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
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
    gl2.xlabel_style = {'size': 14, 'color': '#383838'}
    gl2.ylabel_style = {'size': 14, 'color': '#383838'}
    gl2.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.bottom_labels = False
    gl2.right_labels = False
    gl2.top_labels = True
    gl2.left_labels = True

def create_colormap(color):
    return mcolors.LinearSegmentedColormap.from_list("", ["white", color], N=256)

def normalize_density(density):
    min_val = density.min()
    max_val = density.max()
    return (density - min_val) / (max_val - min_val)

def plot_combined_density(ax, density, phase, color):
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)
    cmap = create_colormap(color)
    norm_density = normalize_density(density)
    
    # Create mask for values below 0.2
    mask = np.ma.masked_where(norm_density < 0.2, norm_density)
    
    cf = ax.contourf(density.lon, density.lat, mask, cmap=cmap, transform=datacrs, extend='max', alpha=0.8)
    ax.coastlines()
    gridlines(ax)
    return cf

def plot_density_for_region(region):
    fig, axes = plt.subplots(nrows=len(PHASES), ncols=len(SEASONS), figsize=(10, 20), subplot_kw={'projection': proj})
    axes = axes.flatten()
    
    for i, phase in enumerate(PHASES):
        for j, season in enumerate(SEASONS):
            ax = axes[i * len(SEASONS) + j]
            season_str = f"_{season}" if season else ""
            infile = os.path.join(INFILES_DIRECTORY, f'{region}_track_density{season_str}.nc')
            
            if not os.path.exists(infile):
                print(f"File not found: {infile}")
                continue

            ds = xr.open_dataset(infile)
            density = ds[phase]

            cf = plot_combined_density(ax, density, phase, COLOR_PHASES[phase])

    cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    colorbar = plt.colorbar(cf, cax=cbar_axes, orientation='horizontal')
    colorbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(hspace=0.3)
    fname = os.path.join(OUTPUT_DIRECTORY, f'{region}_density_map.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Density map for region {region} saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for region in REGIONS:
        plot_density_for_region(region)
