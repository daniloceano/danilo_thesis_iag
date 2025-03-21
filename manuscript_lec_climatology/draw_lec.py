# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    draw_lec.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/03 23:31:13 by daniloceano       #+#    #+#              #
#    Updated: 2025/03/21 00:04:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os

import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pdfs import read_life_cycles

COLOR_PHASES = {
    'Total': '#1d3557',
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'intensification 2': '#ca6702',
    'mature': '#d62828',
    'mature 2': '#9b2226',
    'decay': '#9aa981',
    'decay 2': '#386641',
}

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    from tqdm import tqdm

    files = os.listdir(base_path)

    for filename in tqdm(files, desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]
            try:
                df = pd.read_csv(file_path)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics


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

def plot_boxes(ax, positions, size):
    # Create energy boxes and text labels with updated terms
    for term, pos in positions.items():
        # Draw circles for energy terms
        circle = patches.Circle(
            (pos[0], pos[1]), radius=size / 2, fill=True, color="skyblue", ec="black", linewidth=2, alpha=0.2
        )
        ax.add_patch(circle)

        # Plot the term name in the center of each circle
        ax.text(
            pos[0], pos[1], term, ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )

def write_terms(ax):

    # Termos de conversão
    ax.text(
        -0.25, 0, "Ca", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        0.25, 0, "Ck", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        0, -0.25, "Ce", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        0, 0.25, "Cz", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    
    # Termos de geração/resíduo
    ax.text(
        -0.5, 1.1, "Gz", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        -0.5, -1.1, "Ge", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        0.5, 1.1, "RKz", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        0.5, -1.1, "RKe", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    
    # Termos de fronteira
    ax.text(
        -1.1, 0.5, "BAz", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        -1.1, -0.5, "BAe", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        1.1, 0.5, "BKz", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )
    ax.text(
        1.1, -0.5, "BKe", ha="center", va="center", fontsize=16, color="black", fontweight="bold"
        )


def plot_arrow(ax, start, end, term_value, normalized_value, term, phase):
    """
    Desenha uma seta ou linha curva entre start e end dependendo do tipo de termo.
    Agora a espessura da seta é baseada na intensidade normalizada do fluxo.
    """
    color = COLOR_PHASES[phase]

    # Definir o tamanho da seta
    size = 5 * normalized_value
    if size < 1:
        size = 1

    # Inverter começo e fim se term_value for negativo
    if term_value < 0:
        start, end = end, start

    # Se for um termo de conversão (como "Ca", "Ck"), desenhar uma linha curva
    if term in ["Ca", "Ck", "Ce", "Cz"]:
        # Curvatura baseada no valor do termo
        curvature = 0.2 if term_value > 0 else -0.2
        # Inverter para Cz
        if term == "Cz":
            curvature *= -1

        arrow = FancyArrowPatch(
            start, end, connectionstyle=f"arc3,rad={curvature}",
            arrowstyle='->', color=color, mutation_scale=50, lw=size, alpha=0.9
        )
        ax.add_patch(arrow)
        
    else:
        # Para outros termos, desenhar seta reta
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
                alpha=0.8
            ),
        )

def plot_signal(ax, positions, phase, data, normalized_data, term):
    """
    Plota o sinal "+" ou "-" dentro dos círculos para termos de balanço de energia.
    """

    # Definir o tamanho da seta
    normalized_value = normalized_data[term]
    size = 20 * normalized_value
    if size < 20:
        size = 20
    elif size > 50:
        size = 50

    # Obtenha o valor do termo e defina a cor
    term_value = data[term]
    color = COLOR_PHASES[phase]

    # Colocar offset para cada fase
    offset_value = 0.1
    offset = {
        'incipient': (-offset_value, offset_value),
        'intensification': (offset_value, offset_value),
        'mature': (-offset_value, -offset_value),
        'decay': (offset_value, -offset_value),
    }

    # Verificar se é um termo de balanço de energia
    pos = positions[term]

    # Definir o símbolo "+" ou "-"
    symbol = "+" if term_value >= 0 else "-"
    
    # Verificar se o deslocamento está sendo aplicado corretamente
    ax.text(
        pos[0] + offset[phase][0],  # Deslocamento no eixo X
        pos[1] + offset[phase][1],  # Deslocamento no eixo Y
        symbol,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight="bold",
    )

import matplotlib.lines as mlines

def plot_legend(ax):
    """
    Adiciona a legenda de cores para cada fase no gráfico.
    """
    legend_elements = [
        mlines.Line2D([0], [0], color=COLOR_PHASES['incipient'], lw=4, label='Incipient'),
        mlines.Line2D([0], [0], color=COLOR_PHASES['intensification'], lw=4, label='Intensification'),
        mlines.Line2D([0], [0], color=COLOR_PHASES['mature'], lw=4, label='Mature'),
        mlines.Line2D([0], [0], color=COLOR_PHASES['decay'], lw=4, label='Decay'),
    ]
    
    ax.legend(
        handles=legend_elements, 
        loc='upper left',  # Posição do canto superior esquerdo
        fontsize=12,
        title_fontsize=14,
        bbox_to_anchor=(0, -0.055),
        ncol=4
    )

def plot_term_arrows(ax, size, term, data, normalized_data, positions, phase):

    # Obtenha o valor do termo e o valor normalizado
    term_value = data[term]
    normalized_value = normalized_data[term]

    # Definir um valor base para o deslocamento
    base_displacement = 0.1

    # Usar o valor absoluto para determinar o deslocamento (quanto maior o valor, maior o deslocamento)
    max_displacement_factor = 0.01  # Ajuste esse fator conforme necessário para o máximo de deslocamento
    displacement = base_displacement + (np.abs(term_value) * max_displacement_factor)

    # Obtenha o valor absoluto e aplique uma escala logarítmica
    log_displacement = np.log(np.abs(term_value) + 1)  # "+1" para evitar log(0)

    # Escalonar para ajustar o deslocamento máximo
    displacement = base_displacement + log_displacement * max_displacement_factor

    # Deslocamento para as setas de cada fase
    if term not in ["Ge", "RKz", "Ge", "RKe", "Ca", "Ck"]:
        phase_displacement = {
            'incipient': displacement * 1.5,         
            'intensification': displacement * 0.5,   
            'mature': displacement * -0.5,           
            'decay': displacement * -1.5            
        }
    else:
        phase_displacement = {
            'incipient': displacement * -1.5,         
            'intensification': displacement * -0.5,   
            'mature': displacement * 0.5,           
            'decay': displacement * 1.5            
        }

    phase_colors = {
        'incipient': 'blue',
        'intensification': 'orange',
        'mature': 'red',
        'decay': 'green'
    }

    displacement = phase_displacement.get(phase, 0)  # Deslocamento baseado na fase

    # Definir posições e condições para cada termo
    if term == "Cz":
        start = (positions["∂Az/∂t"][0] + size / 2, positions["∂Az/∂t"][1] + displacement)
        end = (positions["∂Kz/∂t"][0] - size / 2, positions["∂Kz/∂t"][1] + displacement)

    elif term == "Ca":
        start = (positions["∂Az/∂t"][0] + displacement, positions["∂Az/∂t"][1] - size / 2)
        end = (positions["∂Ae/∂t"][0] + displacement, positions["∂Ae/∂t"][1] + size / 2)

    elif term == "Ck":
        start = (positions["∂Kz/∂t"][0] + displacement, positions["∂Ke/∂t"][1] + size / 2)
        end = (positions["∂Ke/∂t"][0] + displacement, positions["∂Kz/∂t"][1] - size / 2)

    elif term == "Ce":
        start = (positions["∂Ae/∂t"][0] + size / 2, positions["∂Ke/∂t"][1] + displacement)
        end = (positions["∂Ke/∂t"][0] - size / 2, positions["∂Ae/∂t"][1] + displacement)

    # Plot text for residuals
    elif term == "Gz":
        start = (positions["∂Az/∂t"][0] + displacement, 1)
        end = (positions["∂Az/∂t"][0] + displacement / 2, positions["∂Az/∂t"][1] + size / 2)

    elif term == "Ge":
        start = (positions["∂Ae/∂t"][0] + displacement, -1)
        end = (positions["∂Ae/∂t"][0] + displacement / 2, positions["∂Ae/∂t"][1] - size / 2)

    elif term == "RKz":
        start = (positions["∂Kz/∂t"][0] + displacement, 1)
        end = (positions["∂Kz/∂t"][0] + displacement / 2, positions["∂Kz/∂t"][1] + size / 2)

    elif term == "RKe":
        start = (positions["∂Ke/∂t"][0] + displacement, -1)
        end = (positions["∂Ke/∂t"][0] + displacement / 2, positions["∂Ke/∂t"][1] - size / 2)

    # Plot text for boundaries
    elif term in ["BAz", "BAe"]:
        refered_term = "∂Az/∂t" if term == "BAz" else "∂Ae/∂t"
        start = (-1, positions[refered_term][1] + displacement)
        end = (positions[refered_term][0] - size / 2, positions[refered_term][1] + displacement / 2)

    elif term in ["BKz", "BKe"]:
        refered_term = "∂Kz/∂t" if term == "BKz" else "∂Ke/∂t"
        start = (1, positions[refered_term][1] + displacement)
        end = (positions[refered_term][0] + size / 2, positions[refered_term][1] + displacement / 2)

    # Plot arrows and signals
    plot_arrow(ax, start, end, term_value, normalized_value, term, phase)

    return start, end

def _plotter(
    phase_means,
    normalized_data_not_energy,
    figures_directory,
    plot_example=False,
    app_logger=False,
):
    
    conversions = TERM_DETAILS["conversion"]["terms"]
    residuals = TERM_DETAILS["residuals"]["terms"]
    boundaries = TERM_DETAILS["boundary"]["terms"]
    budget = ["∂Az/∂t", "∂Ae/∂t", "∂Kz/∂t", "∂Ke/∂t"]
    
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

    data = phase_means.iloc[0]
    normalized_data = normalized_data_not_energy.iloc[0]

    plot_boxes(ax, positions, size)
    write_terms(ax)

    for phase, data in phase_means.iterrows():

        # Plot only for first development cycle
        if '2' in phase or 'residual' in phase or 'total' in phase:
            continue

        # Extract the corresponding normalized data for the day
        normalized_data = normalized_data_not_energy.loc[phase]

        # Plot the Lorenz cycle for the day
        for term in conversions + residuals + boundaries + budget:
            print(f"Plotting {term}")

            is_balance_term = term in ["∂Az/∂t", "∂Ae/∂t", "∂Kz/∂t", "∂Ke/∂t"]
            if is_balance_term:
                plot_signal(ax, positions, phase, data, normalized_data, term)
            else:
                plot_term_arrows(ax, size, term, data, normalized_data, positions, phase)

    # Adicionar legenda com cores das fases
    plot_legend(ax)

    figures_subdirectory = os.path.join(figures_directory, "draw_LEC")
    os.makedirs(figures_subdirectory, exist_ok=True)
    figure_path = os.path.join(figures_subdirectory, f"LEC_test.png")
    plt.savefig(figure_path)
    plt.close()
    (
        app_logger.info(f"Lorenz cycle plot saved to {figure_path}")
        if app_logger
        else print(f"Lorenz cycle plot saved to {figure_path}")
    )


def plot_period_means(periods_df, figures_directory):

    # Selecionar apenas fases do primeiro ciclo de vdia
    periods_df = periods_df.loc[['incipient', 'intensification', 'mature', 'decay']]

    # Renome columns by removing "(finite diff.)"
    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))

    # Initialize an empty DataFrame to store period means
    period_means_df = pd.DataFrame()

    # Iterate through each period and calculate means
    for period_name, row in periods_df.iterrows():
        period_mean = periods_df.loc[period_name]
        # Add the mean to the period_means_df DataFrame
        period_means_df = pd.concat([period_means_df, pd.DataFrame(period_mean).transpose()])

    # Normalize data
    df_not_energy_periods = np.abs(
        period_means_df.drop(columns=["Az", "Ae", "Kz", "Ke", 'BΦE', 'BΦZ'])
    )

    normalized_data_log = np.log1p(df_not_energy_periods)

    # Plot period means
    _plotter(period_means_df, normalized_data_log, figures_directory)


def plot_lorenzcycletoolkit(periods_df, figures_directory):

    # Rename columns by removing "(finite diff.)"
    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))
    plot_period_means(periods_df, figures_directory)

if __name__ == "__main__":
    # Test for Reg1-Representative_fixed
    PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
    base_path = f'{PATH}/csv_database_energy_by_periods'
    figures_directory = "./figures/"

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