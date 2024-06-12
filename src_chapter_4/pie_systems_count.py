import os
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
from matplotlib.font_manager import FontProperties

# Function to adjust the brightness of a color
def adjust_brightness(color, factor):
    rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(*[x/255.0 for x in rgb])
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# Define regions, seasons, and colors
regions = ['ARG', 'LA-PLATA', 'SE-BR']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
COLOR_REGIONS = {
    'ARG': '#3e8fc1',
    'LA-PLATA': '#adad38',
    'SE-BR': '#d73027',
}

# Define output directory
output_directory = '../figures_chapter_4/'

# Create a dictionary to store the total counts for each region
region_totals = {region: 0 for region in regions}

# Create lists to store the data for the pie chart
labels = []
sizes = []
colors = []

# Read the data from CSV files and calculate the total counts
for region in regions:
    region_total = 0
    for season in seasons:
        csv_path = f'../results_chapter_4/count_systems/{season}_count_of_systems_{region}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            season_count = df['Total Count'].sum()
            region_total += season_count
            sizes.append(season_count)
            colors.append(adjust_brightness(COLOR_REGIONS[region], factor= 1.2 + 0.2 * seasons.index(season)))
    region_totals[region] = region_total

# Calculate the total number of systems across all regions
total_systems = sum(region_totals.values())

# Create labels for the outer ring (seasons)
for region in regions:
    for season in seasons:
        csv_path = f'../results_chapter_4/count_systems/{season}_count_of_systems_{region}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            season_count = df['Total Count'].sum()
            percentage = (season_count / region_totals[region]) * 100
            labels.append(f'{season} ({percentage:.1f}%)')

# Create labels for the inner ring (regions)
inner_labels = [f'{regions[i]}\n({region_totals[region]} -  {(region_totals[region] / total_systems) * 100:.1f}%)' for i, region in enumerate(regions)]
inner_sizes = [region_totals[region] for region in regions]
inner_colors = [COLOR_REGIONS[region] for region in regions]

# Plot the pie chart
fig, ax = plt.subplots(figsize=(12, 8))

# Outer ring
font_props = FontProperties(weight='bold')
ax.pie(sizes, labels=labels, labeldistance=0.9, colors=colors, radius=1.3, startangle=90, counterclock=False,
       wedgeprops=dict(width=0.3, edgecolor='w'), textprops={'fontproperties': font_props, 'ha': 'center', 'va': 'center'})

# Inner ring
ax.pie(inner_sizes, labels=None, colors=inner_colors, radius=1, startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a circle at the center to turn the pie into a donut
centre_circle = plt.Circle((0, 0), 0.7, color='white', fc='white', linewidth=0)
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

ax.text(0.45, 0, inner_labels[0], va='center', ha='center', fontsize=12) # ARG
ax.text(-0.4, -0.2, inner_labels[1], va='center', ha='center', fontsize=12) # LA-PLATA
ax.text(-0.3, 0.45, inner_labels[2], va='center', ha='center', fontsize=12) # SE-BR

# Add a legend
handles = [plt.Line2D([0], [0], color=COLOR_REGIONS[region], lw=4) for region in regions]
legend_labels = [f'{region}' for region in regions]
ax.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.85, 1), fontsize=12)

plt.tight_layout()

plt.savefig(os.path.join(output_directory, 'pie_systems_count.png'))
print('Pie chart saved to', os.path.join(output_directory, 'pie_systems_count.png'))
plt.close()
