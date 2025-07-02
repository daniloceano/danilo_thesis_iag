import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt

PATH = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/"
results_path = f'{PATH}/csv_database_energy_by_periods'
eofs_path = f'{PATH}/csv_eofs_energetics'
figures_directory = 'figures/LEC_eof/'

phases = ['incipient', 'intensification', 'mature', 'decay']

TERM_DETAILS = {
    "energy": {"terms": ["Az", "Ae", "Kz", "Ke"], "label": "Energy", "unit": "J·m⁻²"},
    "conversion": {
        "terms": ["Cz", "Ca", "Ck", "Ce"],
        "label": "Conversion",
        "unit": "W·m⁻²",
    },
    "boundary": {
        "terms": ["BAz", "BAe", "BKz", "BKe"],
        "label": "Transport across boundaries",
        "unit": "W·m⁻²",
    },
    "budget_diff": {
        "terms": [
            "∂Az/∂t",
            "∂Ae/∂t",
            "∂Kz/∂t",
            "∂Ke/∂t",
        ],
        "label": "Energy budgets (estimated using finite diffs.)",
        "unit": "W·m⁻²",
    },
    "residuals": {
        "terms": ["Gz", "RKz", "Ge", "RKe"],
        "label": "Residuals",
        "unit": "W·m⁻²",
    },
    "generation_dissipation": {
        "terms": ["Gz", "Ge", "Dz", "De"],
        "label": "Generation/Dissipation",
        "unit": "W·m⁻²",
    },
}

def plot_boxes(ax, data, positions, size):
    # Define edge width range
    min_edge_width = 0
    max_edge_width = 5

    # Create energy boxes and text labels with updated terms
    for term, pos in positions.items():
        term_value = data[term]

        # Scale edge width based on normalized value
        edge_width = (
            min_edge_width + (max_edge_width - min_edge_width) * term_value / 10
        )

        # Determine value text color based on term value
        value_text_color = "#386641"  # Dark green for positive values
        if term_value < 0:
            value_text_color = "#ae2012"  # Dark red for negative values

        square = patches.Rectangle(
            (pos[0] - size / 2, pos[1] - size / 2),
            size,
            size,
            fill=True,
            color="skyblue",
            ec="black",
            linewidth=edge_width,
        )
        ax.add_patch(square)

        # Define the term and value parts
        term_text = f"{term}"  # This will always be black
        value_text = f"{term_value:.2f}"  # Color based on term value

        # Set color for the value text based on the value sign
        value_text_color = "#386641"  # Green for positive values
        if term_value < 0:
            value_text_color = "#ae2012"  # Red for negative values

        # Now, plot the term with black color for the term name and the conditional color for the value
        ax.text(
            pos[0],
            pos[1] + 0.07,
            term_text,
            ha="center",
            va="center",
            fontsize=16,
            color="black",  # Always black for the term name
            fontweight="bold",
        )

        # Plot the value with the appropriate color
        ax.text(
            pos[0],
            pos[1] - 0.05,  # Slightly adjust position for the value text to not overlap
            value_text,
            ha="center",
            va="center",
            fontsize=16,
            color=value_text_color,  # Conditional color for the value
            fontweight="bold",
        )

def plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0), plot_example=False):
    # Define term name and value text parts
    term_text = f"{term}"  # Always black for the term name
    if 'G' in term or 'R' in term:
        value_text = f"{term_value:.2f}"  # Value with standard deviation
    else:
        value_text = f"{term_value:.2f}"  # Value with standard deviation

    # Determine text color based on the term value
    value_text_color = "#386641"  # Green for positive values
    if term_value < 0:
        value_text_color = "#ae2012"  # Red for negative values

    # Midpoint for positioning
    mid_point = (
        (start[0] + end[0]) / 2 + offset[0],
        (start[1] + end[1]) / 2 + offset[1],
    )

    # Adjust offsets for specific terms to avoid overlap
    if term in ["Ca", "BAz", "BAe"]:
        offset_x = -0.05
    elif term in ["Ck", "BKz", "BKe"]:
        offset_x = 0.05
    else:
        offset_x = 0

    if term == "Ce":
        offset_y = -0.09
    elif term == "Cz":
        offset_y = 0.01
    else:
        offset_y = 0

    if term not in ["Gz", "RKz", "Ge", "RKe"]:
        term_text_offset_y = 0.11
    elif term in ["Gz", "RKz"]:
        term_text_offset_y = 0.07
    else:
        term_text_offset_y = - 0.07

    x_pos = mid_point[0] + offset_x
    y_pos = mid_point[1] + offset_y

    # Plot the term name in black
    ax.text(
        x_pos,
        y_pos + term_text_offset_y,
        term_text,
        ha="center",
        va="center",
        color="black",  # Always black for the term
        fontsize=16,
        fontweight="bold",
    )

    # Plot the value with the determined color
    ax.text(
        x_pos,
        y_pos,  # Adjust the position slightly to avoid overlap
        value_text,
        ha="center",
        va="center",
        color=value_text_color,  # Color based on the value
        fontsize=16,
        fontweight="bold",
    )


def plot_term_arrows_and_text(ax, size, term, data, positions):
    term_value = data[term]

    arrow_color = "#5C5850"  # Default color

    if term == "Cz":
        start = (positions["∂Az/∂t"][0] + size / 2, positions["∂Az/∂t"][1])
        end = (positions["∂Kz/∂t"][0] - size / 2, positions["∂Kz/∂t"][1])
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.1))

    elif term == "Ca":
        start = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] - size / 2)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(-0.1, 0))

    elif term == "Ck":
        start = (positions["∂Kz/∂t"][0], positions["∂Ke/∂t"][1] + size / 2)
        end = (positions["∂Ke/∂t"][0], positions["∂Kz/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value,  offset=(0.1, 0))

    elif term == "Ce":
        start = (positions["∂Ae/∂t"][0] + size / 2, positions["∂Ke/∂t"][1])
        end = (positions["∂Ke/∂t"][0] - size / 2, positions["∂Ae/∂t"][1])
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, -0.1))

    # Plot text for residuals
    elif term == "Gz":
        start = (positions["∂Az/∂t"][0], 1)
        end = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.2))

    elif term == "Ge":
        start = (positions["∂Ae/∂t"][0], -1)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, -0.2))

    elif term == "RKz":
        start = (positions["∂Kz/∂t"][0], 1)
        end = (positions["∂Kz/∂t"][0], positions["∂Kz/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.2))

    elif term == "RKe":
        start = (positions["∂Ke/∂t"][0], -1)
        end = (positions["∂Ke/∂t"][0], positions["∂Ke/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, -0.2))

    # Plot text for boundaries
    elif term in ["BAz", "BAe"]:
        refered_term = "∂Az/∂t" if term == "BAz" else "∂Ae/∂t"
        start = (-1, positions[refered_term][1])
        end = (positions[refered_term][0] - size / 2, positions[refered_term][1])
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(-0.23, 0))

    elif term in ["BKz", "BKe"]:
        refered_term = "∂Kz/∂t" if term == "BKz" else "∂Ke/∂t"
        start = (1, positions[refered_term][1])
        end = (positions[refered_term][0] + size / 2, positions[refered_term][1])
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0.23, 0))

    if term_value < 0:
        # Swap start and end for negative values
        start_normalized, end_normalized = end, start
    else:
        start_normalized, end_normalized = start, end

    # Plot arrow
    plot_arrow(ax, start_normalized, end_normalized, data[term], color=arrow_color)

    return start, end


def plot_arrow(ax, start, end, term_value, color="#5C5850"):
    """Draws an arrow on the given axes from start to end point."""

    # Determine arrow size based on term value
    if np.abs(term_value) < 1:
        size = 3 + np.abs(term_value)
    elif np.abs(term_value) < 5:
        size = 3 + np.abs(term_value)
    elif np.abs(term_value) < 10:
        size = 3 + np.abs(term_value)
    else:
        size = 15 + np.abs(term_value) * 0.1

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            facecolor=color,
            edgecolor=color,
            width=size,
            headwidth=size * 3,
            headlength=size * 3,
        ),
    )

def _call_plot(data, variance_phase_eof, eof, phase):
    # Prepare data
    conversions = TERM_DETAILS["conversion"]["terms"]
    residuals = TERM_DETAILS["residuals"]["terms"]
    boundaries = TERM_DETAILS["boundary"]["terms"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Define positions and size of energy boxes
    positions = {
        "∂Az/∂t": (-0.5, 0.5),
        "∂Ae/∂t": (-0.5, -0.5),
        "∂Kz/∂t": (0.5, 0.5),
        "∂Ke/∂t": (0.5, -0.5),
    }
    size = 0.4

    # Plot the boxes for energy terms
    plot_boxes(ax, data, positions, size)

    title_text = f"EOF {eof+1}\n{phase}\nExp. Var.: {variance_phase_eof:.2f}%"

    # Add title
    ax.text(
        0,
        0,
        title_text,
        fontsize=16,
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
    )

    # Iterate over conversion, residual, and boundary terms and plot mean 
    for term in conversions + residuals + boundaries:
        plot_term_arrows_and_text(
            ax, size, term, data, positions)


# Adjust the plotter function to handle both mean and standard deviation
def plot_lorenzcycletoolkit_eof(periods_df, explained_variances, eof, figures_directory):

    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))

    for phase, data in periods_df.iterrows():
        variance_phase_eof = explained_variances.loc[phase].loc[eof]
        # Plot the Lorenz cycle for the day
        _call_plot(data, variance_phase_eof, eof, phase)
        plt.tight_layout()

        figure_path = os.path.join(figures_directory, f"LEC_EOF{eof+1}_{phase}.png")
        plt.savefig(figure_path)
        plt.close()

# DataFrame vazio para armazenar todos os resultados
all_eofs_data = pd.DataFrame()

eof_phases = {}
explained_variances_eof_phases = {}
for eof in range(0, 4):  # Para cada EOF, onde o índice é único
    # Criar Dataframe para armazenar os resultados para o EOF atual
    eof_data = pd.DataFrame()
    explained_variances = pd.DataFrame()
    for phase in phases:

        # Leia o primeiro arquivo CSV para obter as colunas
        results = glob(results_path+'/*.csv')
        dummy_result = pd.read_csv(results[0], index_col=0)
        columns = dummy_result.columns

        # Carregue os dados de EOF para a fase
        phase_directory = os.path.join(eofs_path, phase)
        eof_file_phase = os.path.join(phase_directory, f'eofs.csv')
        df = pd.read_csv(eof_file_phase, header=None)

        # Ajuste os nomes das colunas
        df.columns = columns
        df.columns = [col if '∂' not in col else '∂' + col.split('∂')[1].split('/')[0] + '/∂t' for col in df.columns]

        # Carregar a variância explicada
        explained_variance = pd.read_csv(os.path.join(phase_directory, 'variance_fraction.csv'), header=None)
        explained_variances = pd.concat([explained_variances, explained_variance * 100], axis=1)

        # Obtenha os dados para o EOF atual
        idata = df.iloc[eof]

        # Juntar com o DataFrame principal
        eof_data = pd.concat([eof_data, idata], axis=1)
    
    # Armazenar os resultados para o EOF atual
    eof_data = eof_data.T
    eof_data.index = phases
    eof_phases[eof] = eof_data

    # Armazenar a variância explicada para o EOF atual
    explained_variances = explained_variances.T
    explained_variances.index = phases
    explained_variances_eof_phases[eof] = explained_variances

for eof in range(0, 4):
    periods_df = eof_phases[eof]
    explained_variances = explained_variances_eof_phases[eof]
    plot_lorenzcycletoolkit_eof(periods_df, explained_variances, eof, figures_directory)