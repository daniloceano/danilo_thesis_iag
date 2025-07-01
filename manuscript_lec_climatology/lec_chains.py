# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    lec_chains.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/03 23:31:13 by daniloceano       #+#    #+#              #
#    Updated: 2025/07/01 09:31:22 by daniloceano      ###   ########.fr        #
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
    "budRGet_diff": {
        "terms": [
            "∂Az/∂t (finite diff.)",
            "∂Ae/∂t (finite diff.)",
            "∂Kz/∂t (finite diff.)",
            "∂Ke/∂t (finite diff.)",
        ],
        "label": "Energy budRGets (estimated using finite diffs.)",
        "unit": "W·m⁻²",
    },
    "residuals": {
        "terms": ["RGz", "RKz", "RGe", "RKe"],
        "label": "Residuals",
        "unit": "W·m⁻²",
    },
    "RGeneration_dissipation": {
        "terms": ["RGz", "RGe", "Dz", "De"],
        "label": "RGeneration/Dissipation",
        "unit": "W·m⁻²",
    },
    "comparing_RGeneration": {
        "terms": ["RGz", "RGe", "RGz", "RGe"],
        "label": "Comparing RGeneration",
        "unit": "W·m⁻²",
    },
    "comparing_dissipation": {
        "terms": ["RKz", "Dz", "RKe", "De"],
        "label": "Comparing Dissipation",
        "unit": "W·m⁻²",
    },
}

def plot_boxes(ax, data, normalized_data, positions, size, plot_example=False):
    # Define edRGe width range
    min_edRGe_width = 0
    max_edRGe_width = 5

    # Create energy boxes and text labels with updated terms
    for term, pos in positions.items():
        term_value = int(data[term].values[0])

        # RGet normalized value for the term to determine edRGe width
        normalized_value = normalized_data[term]
        # Scale edRGe width based on normalized value
        edRGe_width = (
            min_edRGe_width + (max_edRGe_width - min_edRGe_width) * normalized_value / 10
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
            linewidth=edRGe_width,
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

    term_value = data[term].values[0]

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
        arrow_color = "#800080"  # Roxo
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
        arrow_color = "#008000"  # Verde
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0.1, 0), plot_example=plot_example
        )

    elif term == "Ce":
        start = (positions["∂Ae/∂t"][0] + size / 2, positions["∂Ke/∂t"][1])
        end = (positions["∂Ke/∂t"][0] - size / 2, positions["∂Ae/∂t"][1])
        arrow_color = "#800080"  # Roxo
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
    elif term == "RGz":
        start = (positions["∂Az/∂t"][0], 1)
        end = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] + size / 2)
        plot_term_text_and_value(
            ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example
        )

    elif term == "RGe":
        start = (positions["∂Ae/∂t"][0], -1)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] - size / 2)
        arrow_color = "#FF0000"  # Vermelho
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
    plot_arrow(ax, start_normalized, end_normalized, data[term][0] * 8, color=arrow_color)

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

    # Annotate Baroclinic Instability
    ax.text(
        -1,
        0,
        "Baroclinic\nInstability",
        fontweight="bold",
        fontsize=18,
        ha="center",
        va="center",
        rotation=0,
        color="#800080",
    )

    # Annotate Barotropic Instability
    ax.text(
        1,
        0,
        "Barotropic\nConversion",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=0,
        color="#008000",
    )

    # Annotate Latent Heat Release
    ax.text(
        -0.9,
        -1,
        "Latent Heat\nRelease",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="#FF0000",
    )

    plt.tight_layout()


def plot_lorenzcycletoolkit(periods_df, figures_directory):

    # Rename columns by removing "(finite diff.)"
    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))
    
    df_not_energy_periods = np.abs(
        periods_df.drop(columns=["Az", "Ae", "Kz", "Ke"])
    )

    # Plot the Lorenz cycle for the day
    _call_plot(periods_df, df_not_energy_periods, plot_example=True)

    figure_path = os.path.join(figures_directory, f"LEC_chains.png")
    plt.savefig(figure_path)
    plt.close()

if __name__ == "__main__":
    # Test for Reg1-Representative_fixed
    figures_directory = "./figures/"

    # Define os termos
    terms = [
        "Az", "Ae", "Kz", "Ke",  # Energy terms
        "Cz", "Ca", "Ck", "Ce",  # Conversion terms
        "BAz", "BAe", "BKz", "BKe",  # Boundary terms
        "∂Az/∂t", "∂Ae/∂t", "∂Kz/∂t", "∂Ke/∂t",  # Budget difference terms
        "RGz", "RGe", "RKz", "RKe",  # Residuals
    ]

    # Cria um DataFrame com todos os valores igual a 1 para os termos
    data = {term: [1] for term in terms}

    df = pd.DataFrame(data)

    # Rename index as ""
    df.index = [""]

    # Plot Lorenz cycle
    plot_lorenzcycletoolkit(df, figures_directory)