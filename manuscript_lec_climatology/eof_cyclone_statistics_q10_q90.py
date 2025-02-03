import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic

# Configurar estilo para publicação científica
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Definir tamanhos de fonte
ylabel_fontsize = 18
title_fontsize = 20
tick_labelsize = 16
legend_fontsize = 14

# Definir cores para sazonalidade
season_colors = {
    'JJA': '#5975A4',  # Azul
    'MAM': '#CC8963',  # Amarelo
    'DJF': '#B55D60',  # Vermelho
    'SON': '#5F9E6E'   # Verde
}

# Caminhos para os arquivos
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'

suffixes = ["q90", "q10"]
data = {}

# Criar diretório para salvar as figuras
output_dir = 'figures/eof_statistics_comparison_q10_q90'
os.makedirs(output_dir, exist_ok=True)

# Carregar e processar os dados para q90 e q10
for suffix in suffixes:
    track_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
    pcs_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_{suffix}.csv'
    
    # Carregar os dados
    tracks = pd.read_csv(track_path)
    pcs = pd.read_csv(pcs_path)

    # Converter a coluna 'date' para datetime
    tracks['date'] = pd.to_datetime(tracks['date'])

    # Unir os dados com base no track_id
    merged_data = tracks.merge(pcs[['track_id', 'dominant_eof']], on='track_id')

    # Restringir análise às primeiras 4 EOFs
    merged_data = merged_data[merged_data['dominant_eof'].isin([1, 2, 3, 4])]

    # Determinar a gênese
    genesis_data = merged_data.groupby('track_id').first().reset_index()

    # Contar sistemas por região de gênese e EOF
    region_counts = genesis_data.groupby(['dominant_eof', 'region']).size().reset_index(name='count')
    region_counts['proportion'] = region_counts.groupby('dominant_eof')['count'].transform(lambda x: x / x.sum())
    region_counts['proportion'] *= 100

    # Pivotar os dados para gráfico de barras
    proportion_pivot = region_counts.pivot(index='dominant_eof', columns='region', values='proportion').fillna(0)

    # Contar ocorrências por EOF e estação
    def get_season(month):
        if month in [12, 1, 2]: return 'DJF'
        elif month in [3, 4, 5]: return 'MAM'
        elif month in [6, 7, 8]: return 'JJA'
        elif month in [9, 10, 11]: return 'SON'



    # Contar ocorrências por EOF e estação considerando apenas a gênese
    first_occurrence = merged_data.groupby('track_id').first().reset_index()
    first_occurrence['season'] = first_occurrence['date'].dt.month.apply(get_season)
    seasonal_counts = first_occurrence.groupby(['dominant_eof', 'season']).size().reset_index(name='count')
    seasonal_counts['season'] = pd.Categorical(seasonal_counts['season'], categories=['DJF', 'MAM', 'JJA', 'SON'], ordered=True)
    total_counts_per_eof = seasonal_counts.groupby('dominant_eof')['count'].sum().reset_index(name='total_count')
    seasonal_counts = seasonal_counts.merge(total_counts_per_eof, on='dominant_eof')
    seasonal_counts['frequency'] = (seasonal_counts['count'] / seasonal_counts['total_count']) * 100


    # Estatísticas por EOF
    cyclone_stats = merged_data.groupby(['track_id', 'dominant_eof']).agg(
        max_intensity=('vor42', 'max'),
        duration=('date', lambda x: (x.max() - x.min()).total_seconds() / (3600 * 24))  # Duração em dias
    ).reset_index()

    # **Cálculo da velocidade média de deslocamento**
    track_velocities = []
    for track_id, group in merged_data.groupby("track_id"):
        group = group.sort_values("date")
        distances = [geodesic((group.iloc[i - 1]["lat vor"], group.iloc[i - 1]["lon vor"]), 
                              (group.iloc[i]["lat vor"], group.iloc[i]["lon vor"])).km
                     for i in range(1, len(group))]
        durations = [(group.iloc[i]["date"] - group.iloc[i - 1]["date"]).total_seconds() / 3600 
                     for i in range(1, len(group))]
        speeds = [d / t if t > 0 else 0 for d, t in zip(distances, durations)]
        if speeds:
            track_velocities.append({"track_id": track_id, "dominant_eof": group["dominant_eof"].iloc[0], "mean_speed": np.mean(speeds)})

    speed_df = pd.DataFrame(track_velocities)

    # Salvar dados processados
    data[suffix] = {
        "proportion_pivot": proportion_pivot,
        "seasonal_counts": seasonal_counts,
        "cyclone_stats": cyclone_stats,
        "speed_stats": speed_df
    }

# **Painel 2x2: Comparação de q90 vs q10 (Região de gênese e sazonalidade)**
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

for i, suffix in enumerate(suffixes):
    # **Gráfico 1: Proporção por região de gênese**
    proportion_melted = data[suffix]["proportion_pivot"].reset_index().melt(id_vars='dominant_eof', var_name='region', value_name='proportion')
    sns.barplot(data=proportion_melted, x='dominant_eof', y='proportion', hue='region', palette="deep", ax=axes[i, 0])

    if suffix == "q90":
        title_label = "EOF(+)"
    elif suffix == "q10":
        title_label = "EOF(-)"
    
    axes[i, 0].set_title(f'({chr(65 + i*2)}) Genesis Proportion - {title_label}', fontsize=title_fontsize, fontweight='bold')
    axes[i, 0].set_xlabel("EOF", fontsize=ylabel_fontsize)  # **Adicionar rótulo no eixo X**
    axes[i, 0].set_ylabel("Proportion (%)", fontsize=ylabel_fontsize)  # **Ajustar rótulo do eixo Y**
    axes[i, 0].tick_params(axis='x', labelsize=tick_labelsize)
    axes[i, 0].tick_params(axis='y', labelsize=tick_labelsize)

    # **Gráfico 2: Ocorrência sazonal**
    sns.barplot(data=data[suffix]["seasonal_counts"], x='dominant_eof', y='frequency', hue='season', palette=season_colors, ax=axes[i, 1])

    axes[i, 1].set_title(f'({chr(65 + i*2 + 1)}) Seasonal Occurrences - {title_label}', fontsize=title_fontsize, fontweight='bold')
    axes[i, 1].set_xlabel("EOF", fontsize=ylabel_fontsize)  # **Adicionar rótulo no eixo X**
    axes[i, 1].set_ylabel("Frequency (%)", fontsize=ylabel_fontsize)  # **Ajustar rótulo do eixo Y**
    axes[i, 1].tick_params(axis='x', labelsize=tick_labelsize)
    axes[i, 1].tick_params(axis='y', labelsize=tick_labelsize)

    # **Manter a legenda apenas no último gráfico**
    if i == 1:
        axes[i, 0].legend(title=None, fontsize=legend_fontsize, bbox_to_anchor=(0.2, -0.15), loc="upper left", ncol=3)
        axes[i, 1].legend(title=None, fontsize=legend_fontsize, bbox_to_anchor=(0.2, -0.15), loc="upper left", ncol=4)
    else:
        axes[i, 0].legend_.remove()
        axes[i, 1].legend_.remove()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_2x2_q90_vs_q10.png'), dpi=300)
plt.close()

# **Correção: Criar dataframe para a contagem de ciclones**
cyclone_count_df = pd.concat([
    data["q90"]["cyclone_stats"].groupby("dominant_eof").size().reset_index(name="cyclone_count").assign(q="q90"),
    data["q10"]["cyclone_stats"].groupby("dominant_eof").size().reset_index(name="cyclone_count").assign(q="q10")
])

# **Substituir as chaves "q90" → "EOF(+)" e "q10" → "EOF(-)" no dicionário data**
data["EOF(+)"] = data.pop("q90")
data["EOF(-)"] = data.pop("q10")

metrics = {
    "cyclone_count": cyclone_count_df,
    "max_intensity": pd.concat([data["EOF(+)"]["cyclone_stats"].assign(q="EOF(+)"),
                                data["EOF(-)"]["cyclone_stats"].assign(q="EOF(-)")]),
    "duration": pd.concat([data["EOF(+)"]["cyclone_stats"].assign(q="EOF(+)"),
                           data["EOF(-)"]["cyclone_stats"].assign(q="EOF(-)")]),
    "mean_speed": pd.concat([data["EOF(+)"]["speed_stats"].assign(q="EOF(+)"),
                             data["EOF(-)"]["speed_stats"].assign(q="EOF(-)")])
}

# **Garantir que a coluna 'q' tenha os valores corrigidos**
for key in ["cyclone_count", "max_intensity", "duration", "mean_speed"]:
    metrics[key]["q"] = metrics[key]["q"].replace({"q90": "EOF(+)", "q10": "EOF(-)"})

# **Painel 2x2: Comparação de EOF(+) vs EOF(-)**
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Definir a paleta de cores com rótulos corrigidos
custom_palette = {"EOF(+)": season_colors['DJF'], "EOF(-)": season_colors['JJA']}

# Loop para gerar cada subplot
for i, (var, df) in enumerate(metrics.items()):
    ax = axes[i // 2, i % 2]

    if var == "cyclone_count":
        # **Gráfico de barras com contagem de ciclones**
        sns.barplot(data=df, x="dominant_eof", y=var, hue="q", palette=custom_palette, ax=ax)
        
        # **Adicionar rótulos numéricos acima das barras**
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',  # Número exato de ciclones
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Posição centralizada
                ha='center', va='bottom', fontsize=14, fontweight='bold'
            )

    else:
        # **Boxplot para as demais variáveis**
        sns.boxplot(data=df, x="dominant_eof", y=var, hue="q", palette=custom_palette, ax=ax)

    # **Configuração de título, eixos e rótulos**
    ax.set_title(f'({chr(65 + i)}) {var.replace("_", " ").title()} EOF(+) vs EOF(-)', 
                 fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel("EOF", fontsize=ylabel_fontsize)
    ax.set_ylabel(var.replace("_", " ").title(), fontsize=ylabel_fontsize)
    ax.tick_params(axis='x', labelsize=tick_labelsize)
    ax.tick_params(axis='y', labelsize=tick_labelsize)

    # **Manter a legenda apenas no primeiro gráfico e corrigir os rótulos e cores**
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ["EOF(+)", "EOF(-)"], title=None, fontsize=legend_fontsize, loc="upper right")
    else:
        ax.legend_.remove()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_2x2_metrics_q90_vs_q10.png'), dpi=300)
plt.close()
