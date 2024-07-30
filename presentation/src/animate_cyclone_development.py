import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cmocean.cm as cmo
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from datetime import timedelta
import cdsapi
import math
import metpy.calc as mpcalc
from metpy.calc import vorticity
from metpy.units import units
import imageio.v2 as imageio

# Constants
COLORS = ["#3B95BF", "#87BF4B", "#BFAB37", "#BF3D3B", "#873e23", "#A13BF0"]
MARKERS = ["s", "o", "^", "v", "<", ">"]
MARKER_COLORS = ["#59c0f0", "#b0fa61", "#f0d643", "#f75452", "#f07243", "#bc6ff7"]
LINESTYLE = "-"
LINEWIDTH = 3
TEXT_COLOR = "#383838"
MARKER_EDGE_COLOR = "grey"
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 18
crs_longlat = ccrs.PlateCarree()

def setup_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle="-", alpha=0.8, color=TEXT_COLOR, linewidth=0.25)
    gl.xlabel_style = {"size": 14, "color": TEXT_COLOR}
    gl.ylabel_style = {"size": 14, "color": TEXT_COLOR}
    gl.bottom_labels = None
    gl.right_labels = None

def map_borders(ax):
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor="none"))
    states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces_lines")
    ax.add_feature(states, edgecolor="#283618", linewidth=1)
    cities = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="populated_places")
    ax.add_feature(cities, edgecolor="#283618", linewidth=1)
    countries = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_0_countries")
    ax.add_feature(countries, edgecolor="black", linewidth=1)

def plot_variable(ax, variable, lat, lon, hgt=None, center_lon=None, center_lat=None):
    if variable.min() < 0:
        norm = colors.TwoSlopeNorm(vmin=variable.min(), vcenter=0, vmax=variable.max())
        cmap = cmo.balance
    else:
        norm = colors.Normalize(vmin=variable.min(), vmax=variable.max())
        cmap = 'turbo'
    
    cf1 = ax.contourf(lon, lat, variable, cmap=cmap, norm=norm, levels=51, transform=crs_longlat)
    # plt.colorbar(cf1, orientation="vertical", shrink=0.5)
    if hgt is not None:
        cs = ax.contour(lon, lat, hgt, levels=11, colors="#344e41", linestyles="dashed", linewidths=1.3, transform=crs_longlat)

    if center_lon is not None and center_lat is not None:
        ax.set_extent([center_lon - 7.5, center_lon + 7.5, center_lat - 7.5, center_lat + 7.5], crs=crs_longlat)

def get_data_cdsapi(track, nc_file):
    min_lat, max_lat = track["lat"].min(), track["lat"].max()
    min_lon, max_lon = track["lon"].min(), track["lon"].max()
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}"

    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]

    track_datetime_index = pd.DatetimeIndex(track.index)
    last_track_timestamp = track_datetime_index.max()
    last_possible_data_timestamp_for_day = pd.Timestamp(f"{last_track_timestamp.strftime('%Y-%m-%d')} 21:00:00")
    need_additional_day = last_track_timestamp > last_possible_data_timestamp_for_day

    dates = track_datetime_index.strftime("%Y%m%d").unique()
    if need_additional_day:
        additional_day = (last_track_timestamp + timedelta(days=1)).strftime("%Y%m%d")
        dates = np.append(dates, additional_day)

    time_range = f"{dates[0]}/{dates[-1]}"
    time_step = str(int((track['date'].iloc[1] - track['date'].iloc[0]).total_seconds() / 3600))
    time_step = "3" if time_step < "3" else time_step

    c = cdsapi.Client(timeout=600)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "pressure_level": 850,
            "date": time_range,
            "area": area,
            "time": f"00/to/23/by/{time_step}",
            "variable": variables,
        },
        nc_file
    )

def create_plots_for_gif(track, nc_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds = xr.open_dataset(nc_file)
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')
    t = ds['t'] * units('K')
    hgt = ds['z'] * units('m')

    zeta = vorticity(u, v)
    potential_temperature = mpcalc.potential_temperature(850 * units('hPa'), t)

    for i, time in enumerate(ds.time):
        itime = pd.Timestamp(time.values).strftime("%Y-%m-%d %H:%M:%S")
        center_lon = track[track['date'] == itime]['lon']
        center_lat = track[track['date'] == itime]['lat']

        if center_lon.empty or center_lat.empty:
            continue
        else:
            center_lon = center_lon.values[0]
            center_lat = center_lat.values[0]

        variable = potential_temperature.isel(time=i).sel(latitude=slice(center_lat + 7.5, center_lat - 7.5),
                                                          longitude=slice(center_lon - 7.5, center_lon + 7.5))
        ihgt = hgt.isel(time=i).sel(latitude=slice(center_lat + 7.5, center_lat - 7.5), longitude=slice(center_lon - 7.5, center_lon + 7.5))
        lat, lon = variable.latitude, variable.longitude

        print(f"Plotting frame {i} at {itime}")
        plt.close("all")
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=crs_longlat)
        fig.add_axes(ax)
        ax.set_global()
        plot_variable(ax, variable.values, lat.values, lon.values, ihgt.values, center_lon, center_lat)
        ax.coastlines(zorder=1)
        map_borders(ax)
        setup_gridlines(ax)
        ax.text(0.5, 1.05, f"Time: {itime}", fontsize=TITLE_FONT_SIZE, transform=ax.transAxes, color=TEXT_COLOR, ha="center", va="bottom")
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9)
        output_file = os.path.join(output_dir, f"frame_{i:03d}.png")
        plt.savefig(output_file)
        plt.close(fig)

def create_gif(output_dir, gif_name):
    images = []
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=100)

def main():
    track_file = '../results_chapter_5/database_tracks/track_periods_energetics_19790820.csv'
    track = pd.read_csv(track_file)
    track['date'] = pd.to_datetime(track['date'])
    track_id = track["track_id"].unique()[0]
    nc_file = f"{track_id}_ERA5.nc"
    output_dir = "output_frames"
    gif_name = "theta_animation.gif"

    if not os.path.exists(nc_file):
        get_data_cdsapi(track, nc_file)

    create_plots_for_gif(track, nc_file, output_dir)
    create_gif(output_dir, gif_name)

if __name__ == "__main__":
    main()