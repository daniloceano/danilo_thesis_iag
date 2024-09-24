# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_LEC.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/03 23:31:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/09/24 19:19:43 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfs import read_life_cycles

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
            "∂Az/∂t (finite diff.)",
            "∂Ae/∂t (finite diff.)",
            "∂Kz/∂t (finite diff.)",
            "∂Ke/∂t (finite diff.)",
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
    "comparing_generation": {
        "terms": ["Gz", "Ge", "Gz", "Ge"],
        "label": "Comparing Generation",
        "unit": "W·m⁻²",
    },
    "comparing_dissipation": {
        "terms": ["RKz", "Dz", "RKe", "De"],
        "label": "Comparing Dissipation",
        "unit": "W·m⁻²",
    },
}

def plot_boxes(ax, data, normalized_data, positions, size, plot_example=False):
    # Define edge width range
    min_edge_width = 0
    max_edge_width = 5

    # Create energy boxes and text labels with updated terms
    for term, pos in positions.items():
        term_value = data[term]

        # Get normalized value for the term to determine edge width
        normalized_value = normalized_data[term]
        # Scale edge width based on normalized value
        edge_width = (
            min_edge_width + (max_edge_width - min_edge_width) * normalized_value / 10
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

        # Term text in bold black
        if plot_example:
            ax.text(
                pos[0],
                pos[1],
                f"{term}",
                ha="center",
                va="center",
                fontsize=16,
                color="k",
                fontweight="bold",
            )

        # Value text in the specified color
        else:
            ax.text(
                pos[0],
                pos[1],
                f"{term_value:.2f}",
                ha="center",
                va="center",
                fontsize=16,
                color=value_text_color,
                fontweight="bold",
            )


def plot_arrow(ax, start, end, term_value, color="#5C5850"):
    """Draws an arrow on the given axes from start to end point."""

    # Determine arrow size based on term value
    for n in range(0, 10):
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


def plot_term_text_and_value(
    ax, start, end, term, term_value, offset=(0, 0), plot_example=False
):
    # Determine text color based on term value
    text_color = "#386641"
    if term_value < 0:
        text_color = "#ae2012"

    mid_point = (
        (start[0] + end[0]) / 2 + offset[0],
        (start[1] + end[1]) / 2 + offset[1],
    )

    if term in ["Ca", "BAz", "BAe"]:
        offset_x = -0.05
    elif term in ["Ck", "BKz", "BKe"]:
        offset_x = 0.05
    else:
        offset_x = 0

    if term == "Ce":
        offset_y = -0.05
    elif term == "Cz":
        offset_y = 0.05
    else:
        offset_y = 0

    x_pos = mid_point[0] + offset_x
    y_pos = mid_point[1] + offset_y

    # Plot term text in bold black
    if plot_example:
        ax.text(
            x_pos,
            y_pos,
            term,
            ha="center",
            va="center",
            fontsize=16,
            color="k",
            fontweight="bold",
        )

    # Plot value text in the specified color
    else:
        ax.text(
            x_pos,
            y_pos,
            f"{term_value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=16,
            fontweight="bold",
        )


def plot_term_value(ax, position, value, offset=(0, 0)):
    ax.text(
        position[0] + offset[0],
        position[1] + offset[1],
        f"{value:.2f}",
        ha="center",
        va="center",
        fontsize=16,
    )


def plot_term_arrows_and_text(ax, size, term, data, positions, plot_example=False):

    term_value = data[term]

    arrow_color = "#5C5850"  # Default color

    if term == "Cz":
        start = (positions["∂Az/∂t"][0] + size / 2, positions["∂Az/∂t"][1])
        end = (positions["∂Kz/∂t"][0] - size / 2, positions["∂Kz/∂t"][1])
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.1), plot_example=plot_example
        )

    elif term == "Ca":
        start = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] - size / 2)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(-0.1, 0),
            plot_example=plot_example,
        )

    elif term == "Ck":
        start = (positions["∂Kz/∂t"][0], positions["∂Ke/∂t"][1] + size / 2)
        end = (positions["∂Ke/∂t"][0], positions["∂Kz/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0.1, 0), plot_example=plot_example
        )

    elif term == "Ce":
        start = (positions["∂Ae/∂t"][0] + size / 2, positions["∂Ke/∂t"][1])
        end = (positions["∂Ke/∂t"][0] - size / 2, positions["∂Ae/∂t"][1])
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(0, -0.1),
            plot_example=plot_example,
        )

    # Plot text for residuals
    elif term == "Gz":
        start = (positions["∂Az/∂t"][0], 1)
        end = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example
        )

    elif term == "Ge":
        start = (positions["∂Ae/∂t"][0], -1)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(0, -0.2),
            plot_example=plot_example,
        )

    elif term == "RKz":
        start = (positions["∂Kz/∂t"][0], 1)
        end = (positions["∂Kz/∂t"][0], positions["∂Kz/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example
        )

    elif term == "RKe":
        start = (positions["∂Ke/∂t"][0], -1)
        end = (positions["∂Ke/∂t"][0], positions["∂Ke/∂t"][1] - size / 2)
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(0, -0.2),
            plot_example=plot_example,
        )

    # Plot text for boundaries
    elif term in ["BAz", "BAe"]:
        refered_term = "∂Az/∂t" if term == "BAz" else "∂Ae/∂t"
        start = (-1, positions[refered_term][1])
        end = (positions[refered_term][0] - size / 2, positions[refered_term][1])
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(-0.23, 0),
            plot_example=plot_example,
        )

    elif term in ["BKz", "BKe"]:
        refered_term = "∂Kz/∂t" if term == "BKz" else "∂Ke/∂t"
        start = (1, positions[refered_term][1])
        end = (positions[refered_term][0] + size / 2, positions[refered_term][1])
        plot_term_text_and_value(
            ax,
            start,
            end,
            term,
            term_value,
            offset=(0.23, 0),
            plot_example=plot_example,
        )

    if term_value < 0:
        # Swap start and end for negative values
        start_normalized, end_normalized = end, start
    else:
        start_normalized, end_normalized = start, end

    # Plot arrow
    plot_arrow(ax, start_normalized, end_normalized, data[term], color=arrow_color)

    return start, end


def _call_plot(data, normalized_data, plot_example=False):
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

    plot_boxes(ax, data, normalized_data, positions, size, plot_example)

    # Add title
    if not plot_example:
        if isinstance(data.name, pd.Timestamp):
            data.name = data.name.strftime("%Y-%m-%d")
        ax.text(
            0,
            0,
            data.name,
            fontsize=16,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )

    for term in conversions + residuals + boundaries:
        start, end = plot_term_arrows_and_text(
            ax, size, term, data, positions, plot_example=plot_example
        )

    plt.tight_layout()


def _plotter(
    daily_means,
    normalized_data_not_energy,
    figures_directory,
    plot_example=False,
    app_logger=False,
):
    if plot_example:
        _call_plot(
            daily_means.iloc[0],
            normalized_data_not_energy.iloc[0],
            plot_example=plot_example,
        )
        figure_name = "example"
        figures_subdirectory = os.path.join(figures_directory, "LEC")
        os.makedirs(figures_subdirectory, exist_ok=True)
        figure_path = os.path.join(figures_subdirectory, f"LEC_{figure_name}.png")
        plt.savefig(figure_path)
        plt.close()
        (
            app_logger.info(f"Lorenz cycle plot saved to {figure_path}")
            if app_logger
            else print(f"Lorenz cycle plot saved to {figure_path}")
        )

    else:
        for date, data in daily_means.iterrows():
            # Extract the corresponding normalized data for the day
            normalized_data = normalized_data_not_energy.loc[date]

            # Plot the Lorenz cycle for the day
            _call_plot(data, normalized_data, plot_example=plot_example)

            if isinstance(data.name, pd.Timestamp):
                figure_name = data.name.strftime("%Y-%m-%d")
            else:
                figure_name = data.name

            figures_subdirectory = os.path.join(figures_directory, "LEC")
            os.makedirs(figures_subdirectory, exist_ok=True)
            figure_path = os.path.join(figures_subdirectory, f"LEC_{figure_name}.png")
            plt.savefig(figure_path)
            plt.close()
            (
                app_logger.info(f"Lorenz cycle plot saved to {figure_path}")
                if app_logger
                else print(f"Lorenz cycle plot saved to {figure_path}")
            )


def plot_period_means(periods_df, figures_directory):

    # Initialize an empty DataFrame to store period means
    period_means_df = pd.DataFrame()

    # Iterate through each period and calculate means
    for period_name, row in periods_df.iterrows():
        period_mean = periods_df.loc[period_name]
        # Add the mean to the period_means_df DataFrame
        period_means_df = pd.concat([period_means_df, pd.DataFrame(period_mean).transpose()])

    # Normalize data
    df_not_energy_periods = np.abs(
        period_means_df.drop(columns=["Az", "Ae", "Kz", "Ke"])
    )
    normalized_data_not_energy_periods = (
        df_not_energy_periods - df_not_energy_periods.min().mean()
    ) / (df_not_energy_periods.max().max() - df_not_energy_periods.min().min())
    normalized_data_not_energy_periods = normalized_data_not_energy_periods.clip(
        lower=1.5, upper=15
    )

    # Plot period means
    _plotter(period_means_df, normalized_data_not_energy_periods, figures_directory)


def plot_lorenzcycletoolkit(periods_df, figures_directory):

    # Rename columns by removing "(finite diff.)"
    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))

    # Normalize data
    df_not_energy = np.abs(periods_df.drop(columns=["Az", "Ae", "Kz", "Ke"]))
    normalized_data_not_energy = (
        (df_not_energy - df_not_energy.min().min())
        / (df_not_energy.max().max() - df_not_energy.min().min())
    ) * 50
    normalized_data_not_energy = normalized_data_not_energy.clip(lower=1.5, upper=15)

    plot_period_means(periods_df, figures_directory)


if __name__ == "__main__":
    # Test for Reg1-Representative_fixed
    PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
    base_path = f'{PATH}/csv_database_energy_by_periods'
    figures_directory = "./figures/lec/"

    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['BA', 'BK'],
        'Pressure Work Terms': ['BΦ'],
        'Generation/Residual Terms': ['G', 'R'],
        'Budget Terms': ['∂']
    }

    terms_prefix = list(groups.keys())

    # Read results
    systems_energetics = read_life_cycles(base_path)

    # Concatenate all systems' dataframes while retaining the system id and phase
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
    all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)
    
    # Convert relevant columns to numeric, forcing errors to NaN
    relevant_columns = ['system_id'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    all_data[relevant_columns] = all_data[relevant_columns].apply(pd.to_numeric, errors='coerce')
    
    # Compute mean across all phases for each system
    mean_data = all_data.drop('Phase', axis=1).groupby('system_id').mean().reset_index()

    # Compute mean across all systems for each phase
    mean_data_by_phase = all_data.drop('system_id', axis=1).groupby('Phase').mean()

    # Reset index to move 'Phase' from being a regular column to an index
    mean_data_by_phase.reset_index(inplace=True)

    # Set 'Phase' as the index again
    mean_data_by_phase.set_index('Phase', inplace=True)

    # Get mean values for all phases combined
    mean_data_all = mean_data_by_phase.mean(axis=0)

    # Convert mean_data_all (which is a Series) to a DataFrame
    mean_data_all_df = mean_data_all.to_frame().T

    # Add a new index for the 'total' row
    mean_data_all_df.index = ['total']

    # Concatenate the mean_data_by_phase with mean_data_all_df
    periods_df = pd.concat([mean_data_by_phase, mean_data_all_df])

    # Plot Lorenz cycle
    plot_lorenzcycletoolkit(periods_df, figures_directory)