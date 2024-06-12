# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    residual_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/31 08:54:11 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/08 17:11:50 by daniloceano      ###   ########.fr        #
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

import cartopy.crs as ccrs
import cmocean as cmo

from metpy.calc import vorticity
from metpy.constants import g

from cyclophaser import determine_periods
from cyclophaser.determine_periods import periods_to_dict, process_vorticity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration variables
TRACKS_DIRECTORY = "/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/processed_tracks_with_periods/"
STUDY_CASE = 19820697 #19920876
CRS = ccrs.PlateCarree() 
OUTPUT_DIRECTORY = './'

# Define the custom colormap
colors_cmap = ['#3555b4', '#547bbd', '#95c9e1', '#d0f0f3',
               '#fdfced',
               '#ffd697', '#fba873', '#f8785b', '#da1a32']
cmap_name = "custom_relative_vorticity"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_cmap, N=9)

color_dots = ['#d62828', '#9aa981', 'gray']
labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]

def get_cdsapi_data(track, infile) -> xr.Dataset:
    """
    Retrieves weather data from the Copernicus Climate Data Store for a given track.
    Args:
        track (pd.DataFrame): The track data for the weather event.
        infile (str): The name of the file to save the downloaded data.
    Returns:
        xr.Dataset: The retrieved dataset.
    """
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

    pressure_levels = ['850']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "geopotential"]
    
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
    """
    Adds coastlines and gridlines to the map.
    Args:
        ax (matplotlib.axes.Axes): The axes object to decorate.
    """
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.7,
                      linewidth=0.5, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = False
    gl.right_labels = False

def draw_box_map(ax, zeta, hgt, lat, lon, norm, levels):
    """
    Plots vorticity and geopotential height on the map.
    Args:
        ax (matplotlib.axes.Axes): The axes object for plotting.
        zeta (xr.DataArray): Vorticity data.
        lat (xr.DataArray): Latitude array.
        lon (xr.DataArray): Longitude array.
        hgt (xr.DataArray): Geopotential height data.
        norm (matplotlib.colors.Normalize): Normalization for color scale.
    """

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

def plot_all_periods(phases_dict, vorticity, ax, idx):
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
        # Extract the base phase name (without suffix)
        base_phase = phase.split()[0]

        # Access the color based on the base phase name
        color = colors_phases[base_phase]

        # Fill between the start and end indices with the corresponding color
        ax.fill_between(vorticity.time, vorticity.zeta.values,
                         where=(vorticity.time >= start) & (vorticity.time <= end + dt),
                        alpha=0.5, color=color, label=base_phase)

    ax.legend(loc='upper right', bbox_to_anchor=(1.65, 1.1), fontsize=14)

    ax.text(0.85, 0.84, labels[idx], fontsize=16, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim(vorticity.time.min(), vorticity.time.max())
    ax.set_ylim(vorticity.zeta.min() - 0.25e-5, 0)

    # Add this line to set x-tick locator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

def process_periods(track):

    zeta_series = (-track['vor42']).to_list()
    x = pd.to_datetime(track['date']).to_list()

    options = {
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing": len(zeta_series) // 12 | 1,
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
    """
    Main function to execute the weather data analysis and visualization.
    """
    track_file = glob(f"{TRACKS_DIRECTORY}/*{str(STUDY_CASE)[:4]}*.csv")
    try:
        tracks = pd.concat([pd.read_csv(f) for f in track_file])
    except Exception as e:
        logging.error(f"Error reading track files: {e}")
        return

    track = tracks[tracks['track_id'] == STUDY_CASE]
    infile = f"{STUDY_CASE}.nc"

    if not os.path.exists(infile):
        try:
            ds = get_cdsapi_data(track, infile)
        except FileNotFoundError as e:
            logging.error(e)
            return
    else:
        ds = xr.open_dataset(infile)

    # Data processing and visualization
    ds = ds.sel(time=slice(track['date'].min(), track['date'].max()))
    lat, lon = ds['latitude'], ds['longitude']
    u_850, v_850 = ds['u'], ds['v']
    zeta_850 = vorticity(u_850, v_850).metpy.dequantify() * 1e4

    # Determine the overall color normalization range and define 9 levels
    zeta_min = zeta_850.min() / 2
    zeta_max = zeta_850.max() / 2
    norm = colors.TwoSlopeNorm(vmin=zeta_min, vcenter=0, vmax=zeta_max)
    levels = np.linspace(zeta_min, zeta_max, 9)  # 9 distinct levels for the colorbar

    plt.close('all')
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.65], width_ratios=[1, 1], right=0.8)

    dates_of_interest = pd.to_datetime(["1982-08-08 12:00:00", "1982-08-09 06:00:00", "1982-08-10 00:00:00", "1982-08-11 18:00:00"])

    for i, time in enumerate(dates_of_interest):
        current_time = pd.to_datetime(time)
        zeta_850_time = zeta_850.sel(time=current_time)
        hgt_850_time = ds['z'].sel(time=current_time) / g

        ax = fig.add_subplot(gs[i//2, i%2], projection=CRS)
        cf1 = draw_box_map(ax, zeta_850_time, hgt_850_time, lat, lon, norm, levels)
        
        # Manually create an axis for the colorbar
        cb_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.06, ax.get_position().width, 0.02])
        cbar = plt.colorbar(cf1, cax=cb_ax, orientation='horizontal', extend='both')
        # cbar.set_label('Relative Vorticity [10^-4 s^-1]')
        cbar.ax.tick_params(labelsize=10)

        # Apply custom tick label formatter
        cbar.formatter = FuncFormatter(round_tick_label)
        cbar.update_ticks()
        
        itrack = track[track['date'] == str(current_time)]
        ax.scatter(itrack['lon vor'], itrack['lat vor'], c='r', marker='o', s=50, zorder=3)
        
        timestr = current_time.strftime("%Y-%m-%d %H:%M")
        ax.set_title(timestr, fontsize=14)
        ax.text(0.85, 1.03, labels[i], fontsize=16, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)


    # Add the fourth subplot without Cartopy projection in the bottom right cell
    ax = fig.add_subplot(gs[2, 0])  # Adjust the indices as per your layout
    df, zeta, periods_dict = process_periods(track)
    plot_all_periods(periods_dict, zeta, ax, 4)
    for date in dates_of_interest:
        ax.plot(date, df[df.index == date]['z'], marker='o', color='r', markersize=10)
        
    out_path = os.path.join(OUTPUT_DIRECTORY, f"residual_study_case.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close('all')

if __name__ == "__main__":
    main()