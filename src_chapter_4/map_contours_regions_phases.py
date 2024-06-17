import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import os
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatches

# Configuration
INFILES_DIRECTORY = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures_chapter_4/track_density_regions/'
PHASES = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2']
REGIONS = ["ARG", "LA-PLATA", "SE-BR"]
AGGREGATE_LABEL = 'Aggregate'

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

LINESTYLES = {
    'LA-PLATA': ':',
    'SE-BR': '--',
    'ARG': '-',
    'default': '-'
}

datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.6, color='#383838', linewidth=0.5)
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

def normalize_density(density):
    min_val = density.min()
    max_val = density.max()
    return (density - min_val) / (max_val - min_val)

def plot_density_contours(ax, density, linestyle, color):
    norm_density = normalize_density(density)
    ax.contour(density.lon, density.lat, norm_density, levels=np.linspace(0.8, 1.2, 2), colors=[color], linestyles=linestyle,
                transform=datacrs, linewidths=4)

def plot_density_for_region(region, combined_density=None):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})
    ax.set_extent([-80, 50, -15, -90], crs=datacrs)

    legend_handles = []

    if combined_densities is not None:
        linestyle = LINESTYLES['default']
    else:
        linestyle = LINESTYLES[region]
    
    for phase in PHASES:
        if region == AGGREGATE_LABEL and combined_density is not None:
            plot_density_contours(ax, combined_density[phase], linestyle, COLOR_PHASES[phase])
        else:
            infile = os.path.join(INFILES_DIRECTORY, f'{region}_track_density.nc')
            if not os.path.exists(infile):
                print(f"File not found: {infile}")
                continue
            ds = xr.open_dataset(infile)
            density = ds[phase]
            plot_density_contours(ax, density, linestyle, COLOR_PHASES[phase])
        legend_handles.append(mpatches.Patch(color=COLOR_PHASES[phase], label=phase))

    ax.coastlines()
    gridlines(ax)
    
    ax.legend(handles=legend_handles, loc='upper right')
    
    fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_{region}.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Density map for region {region} saved in {fname}')

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    combined_densities = {phase: None for phase in PHASES}
    for region in REGIONS:
        plot_density_for_region(region)
        for phase in PHASES:
            infile = os.path.join(INFILES_DIRECTORY, f'{region}_track_density.nc')
            if os.path.exists(infile):
                ds = xr.open_dataset(infile)
                density = ds[phase]
                if combined_densities[phase] is None:
                    combined_densities[phase] = density
                else:
                    combined_densities[phase] += density

    plot_density_for_region(AGGREGATE_LABEL, combined_densities)
