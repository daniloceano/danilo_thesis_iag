# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    eof_export_density_clusters.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2025/02/03 14:25:01 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import xarray as xr
import numpy as np

from glob import glob
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

def compute_density(tracks, num_time):
    k = 64
    longrd = np.linspace(-180, 180, 2 * k)
    latgrd = np.linspace(-87.863, 87.863, k)
    tx, ty = np.meshgrid(longrd, latgrd)
    mesh = np.vstack((ty.ravel(), tx.ravel())).T
    mesh *= np.pi / 180.  # Convert to radians

    pos = tracks[['lat vor', 'lon vor']]
    h = np.vstack([pos['lat vor'].values, pos['lon vor'].values]).T
    h *= np.pi / 180.  # Convert to radians

    kde = KernelDensity(bandwidth=0.05, metric='haversine', kernel='gaussian', algorithm='ball_tree').fit(h)
    v = np.exp(kde.score_samples(mesh)).reshape((k, 2 * k))

    R = 6369345.0 * 1e-3  # Earth radius in km
    factor = (1 / (R ** 2)) * 1.e6
    density = v * pos.shape[0] * factor / num_time

    return density, longrd, latgrd

def export_density_by_cluster(tracks, num_time, output_directory):
    unique_clusters = tracks['cluster'].unique()

    # Remove nan values from unique_clusters
    unique_clusters = unique_clusters[~np.isnan(unique_clusters)]

    for cluster in unique_clusters:
        cluster_tracks = tracks[tracks['cluster'] == cluster]
        print(f"Computing density for cluster {cluster}")

        density, lon, lat = compute_density(cluster_tracks, num_time)
        data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'], name=f"Cluster {cluster}")

        fname = f'{output_directory}/track_density_cluster_{cluster}.nc'
        data.to_netcdf(fname)
        print(f'Wrote {fname}')

def main():
    
    output_directory = f'track_density_clusters'
    os.makedirs(output_directory, exist_ok=True)

    # Get tracks
    tracks_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
    tracks_df = pd.read_csv(tracks_path)

    # Load clusters
    clusters_path = 'figures/eof_clusters_intense/pcs_with_clusters.csv'
    clusters_df = pd.read_csv(clusters_path)

    # Filtrar apenas os track_ids contidos em clusters_df
    filtered_tracks_df = tracks_df[tracks_df['track_id'].isin(clusters_df['track_id'])]

    # Assign clusters to tracks
    filtered_tracks_df = filtered_tracks_df.merge(clusters_df[['track_id', 'cluster']], on='track_id', how='left')
    filtered_tracks_df['cluster'] += 1

    # Filter for unique years and months
    filtered_tracks_df['date'] = pd.to_datetime(filtered_tracks_df['date'])
    unique_years_months = filtered_tracks_df['date'].dt.to_period('M').unique()
    num_time = len(unique_years_months)
    print(f"Total number of time months: {num_time}")

    # Export density maps for each cluster
    export_density_by_cluster(filtered_tracks_df, num_time, output_directory)

if __name__ == '__main__':
    main()
