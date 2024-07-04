# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tracks_ep3_DItMD2.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/03 13:09:34 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/04 16:45:55 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from glob import glob
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

# Configuration constants
datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))
CSV_PATH = '../results_chapter_5/database_tracks'
KMEANS_PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/results_kmeans/all_systems/DItMD2'
OUTPUT_DIRECTORY = '../figures_chapter_6'
LINE_STYLES = {'default': 'solid'}
COLOR_PHASES = {'incipient': 'blue', 'mature': 'green', 'decay': 'red'}
PHASES = ['incipient', 'mature']

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

def plot_density_contours(ax, latitudes, longitudes, color):
    """
    Plot density contours for the given latitude and longitude data.
    """
    # Define the range based on the extent of the map
    range_lon = [-80, 50]
    range_lat = [-90, -15]

    # Create a 2D histogram to estimate density
    density, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=[100, 100], range=[range_lon, range_lat])
    
    # Apply Gaussian filter for smoothing
    density = gaussian_filter(density, sigma=4)

    # Normalize the density
    density = density.T
    density = np.ma.masked_where(density == 0, density)  # Mask zero values
    norm_density = (density - density.min()) / (density.max() - density.min())

    # Plot density contours
    contour = ax.contour(xedges[:-1], yedges[:-1], norm_density, levels=np.linspace(0, 1, 3), 
                          colors=[color], transform=datacrs, linewidths=4)

def plot_complete_track(df, track_id, ax):
    """
    Plot the complete track for a given track_id.
    """
    track_data = df[df['track_id'] == track_id]
    latitudes = track_data['lat'].values
    longitudes = track_data['lon'].values

    ax.plot(longitudes, latitudes, linestyle='-', linewidth=2, transform=datacrs, alpha=0.8)
    gridlines(ax)

def get_cyclone_ids_by_cluster(results_path):
    json_path = glob(f'{results_path}/kmeans_results*.json')[0]
    with open(json_path, 'r') as file:
        cluster_data = json.load(file)
    
    cluster_cyclones = {cluster: data['Cyclone IDs'] for cluster, data in cluster_data.items()}
    return cluster_cyclones

def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Read the list of systems to be analyzed
    selected_systems = get_cyclone_ids_by_cluster(KMEANS_PATH)['Cluster 3']

    # Get track files and select for the selected systems
    track_files = glob(f'{CSV_PATH}/*.csv')
    track_files = [f for f in track_files if int(os.path.basename(f).split('.')[0].split('_')[-1]) in selected_systems]

    # Plot complete tracks for each track_id
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})
    ax.set_extent([-70, -10, -20, -60], crs=datacrs)
    for track_file in track_files:
        # Get track_id
        track_id = int(os.path.basename(track_file).split('.')[0].split('_')[-1])

        # Read the CSV file
        df = pd.read_csv(track_file)

        # Select only the selected systems
        df = df[df['track_id'].isin(selected_systems)]

        # Convert longitude to -180 to 180
        df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360, df['lon'])
        plot_complete_track(df, track_id, ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.3)

    ax.set_title('Complete tracks for Cluster 3 - Life Cycle: DItMD2', fontsize=16)

    fname = os.path.join(OUTPUT_DIRECTORY, f'complete_tracks_cluster_3_DItMD2.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Complete track plot for track_id {track_id} saved in {fname}')

if __name__ == '__main__':
    main()