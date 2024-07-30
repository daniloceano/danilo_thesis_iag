# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    semi_lagrangian_animation.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/29 19:00:08 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/30 09:35:41 by daniloceano      ###   ########.fr        #
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

# First Animation: Semi-Lagrangian with squares appearing at each time step
fig1 = plt.figure(figsize=(10, 8))
ax1 = plt.axes(projection=ccrs.PlateCarree())
ax1.set_extent([min_lon_15, max_lon_15, min_lat_15, max_lat_15])
ax1.coastlines(zorder=1)
map_borders(ax1)
setup_gridlines(ax1)

scat1 = ax1.scatter([], [], color='red', s=50, transform=ccrs.PlateCarree())
line1, = ax1.plot([], [], color='blue', linewidth=2, transform=ccrs.PlateCarree())
rect1 = None  # Initialize the rectangle object

def update1(frame):
    global rect1
    scat1.set_offsets(track[['lon', 'lat']].iloc[:frame + 1].values)
    line1.set_data(track['lon'].iloc[:frame + 1].values, track['lat'].iloc[:frame + 1].values)
    
    current_lon = track['lon'].iloc[frame]
    current_lat = track['lat'].iloc[frame]
    
    if rect1:
        rect1.remove()  # Remove the previous rectangle
    
    rect1 = Rectangle((current_lon - 7.5, current_lat - 7.5), 15, 15,
                      linewidth=1, edgecolor='red', facecolor='none', zorder=3)
    ax1.add_patch(rect1)
    
    return [scat1, line1, rect1]

ani1 = FuncAnimation(fig1, update1, frames=len(track), interval=200, blit=True)

# Second Animation: All squares initially present, turning red with alpha as the track passes
fig2 = plt.figure(figsize=(10, 8))
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.set_extent([min_lon_15, max_lon_15, min_lat_15, max_lat_15])
ax2.coastlines(zorder=1)
map_borders(ax2)
setup_gridlines(ax2)

scat2 = ax2.scatter([], [], color='red', s=50, transform=ccrs.PlateCarree())
line2, = ax2.plot([], [], color='blue', linewidth=2, transform=ccrs.PlateCarree())

# Create all squares initially
squares = {}
for i in range(len(track)):
    current_lon = track['lon'].iloc[i]
    current_lat = track['lat'].iloc[i]
    key = (current_lon - 7.5, current_lat - 7.5)
    rect = Rectangle(key, 15, 15, linewidth=1, edgecolor='red', facecolor='none', zorder=3)
    squares[key] = rect
    ax2.add_patch(rect)

def update2(frame):
    scat2.set_offsets(track[['lon', 'lat']].iloc[:frame + 1].values)
    line2.set_data(track['lon'].iloc[:frame + 1].values, track['lat'].iloc[:frame + 1].values)
    
    # Update squares' face color based on the cyclone's current position
    for key, rect in squares.items():
        if rect.get_xy() == (track['lon'].iloc[frame] - 7.5, track['lat'].iloc[frame] - 7.5):
            rect.set_facecolor('red')
            rect.set_alpha(0.3)
        else:
            rect.set_facecolor('none')
    
    return [scat2, line2] + list(squares.values())

ani2 = FuncAnimation(fig2, update2, frames=len(track), interval=200, blit=True)

# Save the animations
ani1.save('semi_lagrangian_cyclone_path_animation.mp4', writer='ffmpeg')
ani2.save('semi_lagrangian_all_squares_animation.mp4', writer='ffmpeg')

plt.show()
