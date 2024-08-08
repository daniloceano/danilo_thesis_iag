import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Step 1: Loop through all CSV files in the directory
results_dir = "../../results_chapter_5/database_tracks/"
csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

# Function to read and filter a CSV file
def read_and_filter_csv(file):
    df = pd.read_csv(file)
    if {'Ca', 'Ck', 'Ge'}.issubset(df.columns):
        return df[['Ca', 'Ck', 'Ge']]
    return None

# Step 2: Use ThreadPoolExecutor to read files in parallel
data = []
with ThreadPoolExecutor() as executor:
    future_to_file = {executor.submit(read_and_filter_csv, file): file for file in csv_files}
    for future in tqdm(as_completed(future_to_file), total=len(csv_files), desc="Processing files"):
        result = future.result()
        if result is not None:
            data.append(result)

# Step 3: Create a DataFrame from the collected data
combined_df = pd.concat(data, ignore_index=True)

# Optionally, subsample the data to reduce size (uncomment if needed)
# combined_df = combined_df.sample(frac=0.1, random_state=1)

# Step 4: Prepare data for surface plot
x = combined_df['Ca'].values
y = combined_df['Ck'].values
z = combined_df['Ge'].values

# Define a finer grid
xi = np.linspace(min(x), max(x), 200)
yi = np.linspace(min(y), max(y), 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate using griddata with linear method
zi = griddata((x, y), z, (xi, yi), method='linear')

# Step 5: Plot the data in a 3D surface plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot surface in blue
surf = ax.plot_surface(xi, yi, zi, color='blue', edgecolor='none')

ax.set_xlabel('Baroclinic Instability (Ca)')
ax.set_ylabel('Barotropic Instability (Ck)')
ax.set_zlabel('Latent Heat Release (Ge)')
plt.title('3D Surface Plot of Ca, Ck, and Ge')

plt.savefig(f"../..//conclusion_3d.png")
