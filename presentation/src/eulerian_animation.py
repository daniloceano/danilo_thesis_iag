# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    eulerian_animation.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/29 19:00:08 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/29 19:52:01 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Read track data
track_file = '../results_chapter_5/database_tracks/track_periods_energetics_19790820.csv'
track = pd.read_csv(track_file)
track['date'] = pd.to_datetime(track['date'])

# Read additional synthetic track data
synthetic_track_file = './synthetic_track.csv'
synthetic_track = pd.read_csv(synthetic_track_file)
synthetic_track['date'] = pd.to_datetime(synthetic_track['date'])

# Define map boundaries with original 5-degree buffer
min_lon_5, max_lon_5 = track['lon'].min() - 5, track['lon'].max() + 5
min_lat_5, max_lat_5 = track['lat'].min() - 5, track['lat'].max() + 5

# Define map boundaries with 15-degree buffer for the whole domain
min_lon_15, max_lon_15 = track['lon'].min() - 15, track['lon'].max() + 15
min_lat_15, max_lat_15 = track['lat'].min() - 15, track['lat'].max() + 15

def setup_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle="-", alpha=0.8, linewidth=0.25)
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
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

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min_lon_15, max_lon_15, min_lat_15, max_lat_15])
ax.coastlines(zorder=1)
map_borders(ax)
setup_gridlines(ax)

# Add the rectangle with the original 5-degree buffer
rect = Rectangle((min_lon_5, min_lat_5), max_lon_5 - min_lon_5, max_lat_5 - min_lat_5,
                 linewidth=1, edgecolor='red', facecolor='none', zorder=3)
ax.add_patch(rect)

# Initialize scatter plots and lines for both tracks
scat1 = ax.scatter([], [], color='red', s=50, transform=ccrs.PlateCarree())
line1, = ax.plot([], [], color='blue', linewidth=2, transform=ccrs.PlateCarree())

scat2 = ax.scatter([], [], color='green', s=50, transform=ccrs.PlateCarree())
line2, = ax.plot([], [], color='orange', linewidth=2, transform=ccrs.PlateCarree())

def update(frame):
    # Update original track
    scat1.set_offsets(track[['lon', 'lat']].iloc[:frame + 1].values)
    line1.set_data(track['lon'].iloc[:frame + 1].values, track['lat'].iloc[:frame + 1].values)
    
    # Update synthetic track
    scat2.set_offsets(synthetic_track[['lon', 'lat']].iloc[:frame + 1].values)
    line2.set_data(synthetic_track['lon'].iloc[:frame + 1].values, synthetic_track['lat'].iloc[:frame + 1].values)
    
    return scat1, line1, scat2, line2

ani = FuncAnimation(fig, update, frames=max(len(track), len(synthetic_track)), interval=200, blit=True)

# Save the animation
ani.save('eulerian_animation.mp4', writer='ffmpeg')

plt.show()
