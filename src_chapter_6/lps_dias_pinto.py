from lorenz_phase_space.phase_diagrams import Visualizer
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('../results_chapter_6/Reg1-Representative_NCEP-R2_fixed/Reg1-Representative_NCEP-R2_fixed_results.csv')

# Extract relevant columns
ck = data['Ck']
ca = data['Ca']
ge = data['Ge']
ke = data['Ke']
bke = data['BKe']
bae = data['BAe']

# Initialize the Lorenz Phase Space plotter for mixed type without zoom
lps_mixed = Visualizer(
    LPS_type='mixed', zoom=True,
    y_limits=(ca.min() * 1.1, ca.max() * 1.1),
    x_limits=(ck.min() * 1.1, ck.max() * 1.1),
    color_limits=(ge.min() * 1.1, ge.max() * 1.1),
    marker_limits=(ke.min() * 1.1, ke.max() * 1.1)
)

# Plot data for mixed type
lps_mixed.plot_data(x_axis=ck, y_axis=ca, marker_color=ge, marker_size=ke)

# Save the mixed type visualization
fname_mixed = '../figures_chapter_6/lps-mixed_Reg1-Representative_NCEP-R2_fixed'
plt.savefig(f"{fname_mixed}.png", dpi=300)
print(f"Saved {fname_mixed}.png")

# Initialize the Lorenz Phase Space plotter for import type without zoom
lps_import = Visualizer(
    LPS_type='imports', zoom=True,
    y_limits=(bae.min() * 1.1, bae.max() * 1.1),
    x_limits=(bke.min() * 1.1, bke.max() * 1.1),
    color_limits=(ge.min() * 1.1, ge.max() * 1.1),
    marker_limits=(ke.min() * 1.1, ke.max() * 1.1)
)

# Plot data for import type
lps_import.plot_data(x_axis=bke, y_axis=bae, marker_color=ge, marker_size=ke)

# Save the import type visualization
fname_import = '../figures_chapter_6/lps-import_Reg1-Representative_NCEP-R2_fixed'
plt.savefig(f"{fname_import}.png", dpi=300)
print(f"Saved {fname_import}.png")
