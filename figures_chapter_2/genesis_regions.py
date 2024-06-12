import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches


def plot_region_box(ax, coords, edgecolor, label=None):
    for coord in coords:
        lon_corners = np.array([coord[0], coord[2], coord[2], coord[0]])
        lat_corners = np.array([coord[1], coord[1], coord[3], coord[3]])

        poly_corners = np.column_stack((lon_corners, lat_corners))
        polygon = mpatches.Polygon(poly_corners, closed=True, ec=edgecolor,
                                    fill=False, lw=2, alpha=1,
                                      transform=ccrs.Geodetic())
        ax.add_patch(polygon)

        if label:
            text_lat = coord[3]
            text_lon = coord[2] - 1
            if label == 'ARG':
                text_lat = coord[3] - 2
                text_lon = coord[2] + 6
            elif label == 'SE-BR':
                text_lon = coord[2] + 5
                
            ax.text(text_lon, text_lat, label, transform=ccrs.Geodetic(),
                     fontsize=16, color='k', fontweight='bold', ha='right', va='bottom')
            
def create_map_and_axes():
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size to accommodate subplots
    proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -50.0))

    # Create a grid of subplots for seasons (2 rows, 2 columns)
    # axs = fig.subplots(2, 2, subplot_kw={'projection': proj})
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    ax.set_extent([-80, -20, -15, -55], crs=ccrs.PlateCarree())
    
    return fig, ax

def add_gridlines_and_continents(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.8, lw=0.35, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgray', alpha=0.7), zorder=1)

def main():
    fig, ax = create_map_and_axes()
    add_gridlines_and_continents(ax)

    add_gridlines_and_continents(ax)

    regions = {
        "SE-BR": [(-52, -38, -37, -23)],
        "LA-PLATA": [(-69, -38, -52, -23)],
        "ARG": [(-70, -55, -50, -39)],
    }

    for region, coords in regions.items():
        plot_region_box(ax, coords, edgecolor='#383838', label=region)

    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray', facecolor='None')
    ax.coastlines()

    fname = './genesis_regions.png'
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f'Genesis regions saved in {fname}')

if __name__ == "__main__":
    main()