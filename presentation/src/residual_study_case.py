# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    residual_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/31 08:54:11 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/31 16:34:45 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cdsapi
import math
import xarray as xr
import os
import logging
import pandas as pd
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import cartopy.crs as ccrs
import cmocean as cmo

from metpy.calc import vorticity
from metpy.constants import g

from cyclophaser import determine_periods
from cyclophaser.determine_periods import periods_to_dict, process_vorticity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration variables
TRACKS_DIRECTORY = "/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/processed_tracks_with_periods"
STUDY_CASE = 19820697 #19920876
CRS = ccrs.PlateCarree() 
OUTPUT_DIRECTORY = '../animations/'

# Define the custom colormap
colors_cmap = ['#3555b4', '#547bbd', '#95c9e1', '#d0f0f3',
               '#fdfced',
               '#ffd697', '#fba873', '#f8785b', '#da1a32']
cmap_name = "custom_relative_vorticity"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_cmap, N=9)

color_dots = ['#d62828', '#9aa981', 'gray']
labels = ["(A)", "(B)", "(C)", "(D)"]

def get_cdsapi_data(track, infile) -> xr.Dataset:
    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['lat vor'].min(), track['lat vor'].max()
    min_lon, max_lon = track['lon vor'].min(), track['lon vor'].max()

    # Apply a 15-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)

    # Define the area for the request
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}"

    pressure_levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                       '100', '125', '150', '175', '200', '225', '250', '300', '350',
                       '400', '450', '500', '550', '600', '650', '700', '750', '775',
                       '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature",
                 "vertical_velocity", "geopotential"]
    
    # Convert unique dates to string format for the request
    dates = pd.to_datetime(track['date'].tolist())
    start_date = dates[0].strftime("%Y%m%d")
    end_date = dates[-1].strftime("%Y%m%d")
    time_range = f"{start_date}/{end_date}"
    time_step = '3'

    # Log track file bounds and requested data bounds
    logging.info(f"Track File Limits: min_lon: {min_lon}, max_lon: {max_lon}, min_lat: {min_lat}, max_lat: {max_lat}")
    logging.info(f"Buffered Data Bounds: min_lon: {buffered_min_lon}, max_lon: {buffered_max_lon}, min_lat: {buffered_min_lat}, max_lat: {buffered_max_lat}")
    logging.info(f"Requesting data for time range: {time_range}, and time step: {time_step}...")

    # Load ERA5 data
    logging.info("Retrieving data from CDS API...")
    c = cdsapi.Client()
    try:
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "pressure_level": pressure_levels,
                "date": time_range,
                "area": area,
                'time': f'00/to/23/by/{time_step}',
                "variable": variables,
            }, infile
        )
    except Exception as e:
        logging.error(f"Error retrieving data from CDS API: {e}")
        raise

    if not os.path.exists(infile):
        raise FileNotFoundError("CDS API file not created.")
    
    try:
        ds = xr.open_dataset(infile)
    except Exception as e:
        logging.error(f"Error opening dataset: {e}")
        raise

    return ds

def map_decorators(ax):
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.7,
                      linewidth=0.5, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = False
    gl.right_labels = False

def draw_box_map(ax, u, v, zeta, hgt, lat, lon, norm, levels):
    # Subsample the data for a less dense quiver plot
    n = 10  # Subsampling factor (every nth point)
    u_sub = u[::n, ::n]
    v_sub = v[::n, ::n]
    lat_sub = lat[::n]
    lon_sub = lon[::n]

    cmap = cmo.cm.balance
    cf1 = ax.contourf(lon, lat, zeta, cmap=custom_cmap, norm=norm, levels=levels, transform=CRS,
                      extend='both') 
    cs = ax.contour(lon, lat, hgt, levels=np.linspace(hgt.min(), hgt.max(), 11),
                    colors='#383838', linestyles='dashed', linewidths=1.5, transform=CRS)
    ax.clabel(cs, cs.levels, inline=True, fontsize=10, fmt='%1.0f')
    map_decorators(ax)
    return cf1  # Return the contour fill object

def round_tick_label(x, pos):
    return f"{x:.2f}"

def plot_all_periods(phases_dict, vorticity, ax, current_time):
    colors_phases = {'incipient': '#65a1e6',
                      'intensification': '#f7b538',
                        'mature': '#d62828',
                          'decay': '#9aa981',
                          'residual': 'gray'}

    ax.plot(vorticity.time, vorticity.zeta, linewidth=10, color='gray', alpha=0.8, label=r'ζ')

    if len(vorticity.time) < 50:
        dt = pd.Timedelta(1, unit='h')
    else:
       dt = pd.Timedelta(0, unit='h')

    # Shade the areas between the beginning and end of each period
    for phase, (start, end) in phases_dict.items():
        base_phase = phase.split()[0]
        color = colors_phases[base_phase]
        ax.fill_between(vorticity.time, vorticity.zeta.values,
                         where=(vorticity.time >= start) & (vorticity.time <= end + dt),
                        alpha=0.5, color=color, label=base_phase)

    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1), fontsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim(vorticity.time.min(), vorticity.time.max())
    ax.set_ylim(vorticity.zeta.min() - 0.25e-5, 0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    if current_time in vorticity.time.values:
        ax.plot(current_time, vorticity.sel(time=current_time)['zeta'], 'ro', markersize=10)

def process_periods(track):
    zeta_series = (-track['vor42']).to_list()
    x = pd.to_datetime(track['date']).to_list()

    options = {
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing": "auto",
            "use_smoothing_twice": False}
    }

    df = determine_periods(zeta_series, x, **options)
    periods_dict = periods_to_dict(df)

    zeta_df = pd.DataFrame(track['vor42'].rename('zeta'))
    zeta_df.index = pd.to_datetime(track['date']).rename('time')

    zeta = process_vorticity(zeta_df.copy(), **options['process_vorticity_args'])

    for variable in zeta:    
        zeta[variable] = -zeta[variable]

    return df, zeta, periods_dict

def main():
    track_file = glob(f"{TRACKS_DIRECTORY}/*{str(STUDY_CASE)[:4]}*.csv")
    try:
        tracks = pd.concat([pd.read_csv(f) for f in track_file])
    except Exception as e:
        logging.error(f"Error reading track files: {e}")
        return

    track = tracks[tracks['track_id'] == STUDY_CASE]
    infile = f"../data/{STUDY_CASE}.nc"

    if not os.path.exists(infile):
        try:
            ds = get_cdsapi_data(track, infile)
        except FileNotFoundError as e:
            logging.error(e)
            return
    else:
        ds = xr.open_dataset(infile)

    # Data processing
    ds = ds.sel(time=slice(track['date'].min(), track['date'].max()))
    lat, lon = ds['latitude'], ds['longitude']
    u_850, v_850 = ds['u'].sel(level=850), ds['v'].sel(level=850)
    zeta_850 = vorticity(u_850, v_850).metpy.dequantify() * 1e4

    zeta_min = zeta_850.min() / 2
    zeta_max = zeta_850.max() / 2
    norm = colors.TwoSlopeNorm(vmin=zeta_min, vcenter=0, vmax=zeta_max)
    levels = np.linspace(zeta_min, zeta_max, 9)

    fig = plt.figure(figsize=(12, 9.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1], hspace=-0.2)
    ax1 = fig.add_subplot(gs[0, :], projection=CRS)
    ax2 = fig.add_subplot(gs[1, :])

    df, zeta, periods_dict = process_periods(track)
    df_sliced = df[df.index.isin(pd.to_datetime(ds.time.values))]

    def update(frame):
        current_time = pd.to_datetime(df_sliced.index[frame])
        u_850_time, v_850_time = u_850.sel(time=current_time), v_850.sel(time=current_time)
        zeta_850_time = zeta_850.sel(time=current_time)
        hgt_850_time = ds['z'].sel(level=850).sel(time=current_time) / g

        ax1.clear()
        draw_box_map(ax1, u_850_time, v_850_time, zeta_850_time, hgt_850_time, lat, lon, norm, levels)
        ax1.set_title(current_time.strftime("%Y-%m-%d %H:%M"), fontsize=14)
        itrack = track[track['date'] == str(current_time)]
        ax1.scatter(itrack['lon vor'], itrack['lat vor'], c='r', marker='o', s=50, zorder=3)

        ax2.clear()
        plot_all_periods(periods_dict, zeta, ax2, current_time)

    ani = animation.FuncAnimation(fig, update, frames=len(df_sliced), repeat=True)
    ani.save(os.path.join(OUTPUT_DIRECTORY, 'residual_study_case_animation.mp4'), writer='ffmpeg', fps=1)
    print(f'Animation saved in {os.path.join(OUTPUT_DIRECTORY, "residual_study_case_animation.mp4")}')

if __name__ == "__main__":
    main()