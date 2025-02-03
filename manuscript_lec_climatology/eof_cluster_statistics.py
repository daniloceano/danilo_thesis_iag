import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definir cores para clusters e regiões
season_colors = {
    'JJA': '#5975A4',  # Azul
    'MAM': '#CC8963',  # Amarelo
    'DJF': '#B55D60',  # Vermelho
    'SON': '#5F9E6E'   # Verde
}

region_colors = {
    'ARG': season_colors['JJA'],      # Azul para Argentina (JJA)
    'LA-PLATA': season_colors['MAM'], # Amarelo para La Plata (MAM)
    'SE-BR': season_colors['SON']     # Verde para Sudeste do Brasil (SON)
}

# Definir tamanhos de fonte
ylabel_fontsize = 18
title_fontsize = 20
tick_labelsize = 16
legend_fontsize = 14

# Caminho dos arquivos
clusters_path = 'figures/eof_clusters_intense/pcs_with_clusters.csv'
tracks_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
output_dir = "figures/eof_clusters_intense"

# Carregar os dados
clusters_df = pd.read_csv(clusters_path)
tracks_df = pd.read_csv(tracks_path)

# Ajustar numeração dos clusters
clusters_df['cluster'] += 1

# Converter coluna de datas
tracks_df['date'] = pd.to_datetime(tracks_df['date'])

# Calcular a derivada temporal da vorticidade (dvor_dt) e converter para taxa por segundo
tracks_df = tracks_df.sort_values(by=['track_id', 'date'])
tracks_df['dvor_dt'] = tracks_df.groupby('track_id')['vor42'].diff() / 3600  # 1 hora = 3600 segundos

# Filtrar apenas os track_ids contidos em clusters_df
filtered_tracks_df = tracks_df[tracks_df['track_id'].isin(clusters_df['track_id'])]

# Determinar a estação do primeiro registro de cada track_id
filtered_tracks_df['season'] = filtered_tracks_df.groupby('track_id')['date'].transform(lambda x: x.dt.month.map(
    lambda m: 'DJF' if m in [12, 1, 2] else
              'MAM' if m in [3, 4, 5] else
              'JJA' if m in [6, 7, 8] else 'SON'
))

# Obter a primeira ocorrência para cada track_id para identificar a estação e a região de gênese
genesis_info_df = filtered_tracks_df.groupby('track_id').first().reset_index()[['track_id', 'season', 'region']]

# Mesclar com clusters_df para associar cluster, estação e região de gênese
clusters_seasonality_df = clusters_df.merge(genesis_info_df, on='track_id', how='left')

# Contagem de sistemas por cluster
cluster_counts = clusters_df['cluster'].value_counts().sort_index()

# Contagem sazonal por cluster
seasonal_counts = clusters_seasonality_df.groupby(['cluster', 'season']).size().unstack(fill_value=0)

# Contagem de sistemas por região de gênese para cada cluster
genesis_counts = clusters_seasonality_df.groupby(['cluster', 'region']).size().unstack(fill_value=0)

# Intensidade máxima (vor42) dos sistemas por cluster
intensity_df = filtered_tracks_df.groupby('track_id')['vor42'].max().reset_index()
intensity_df = intensity_df.merge(clusters_df[['track_id', 'cluster']], on='track_id', how='left')

# Taxa de crescimento médio (dvor_dt) dos sistemas por cluster durante intensificação
intensification_df = tracks_df[tracks_df['period'] == 'intensification']
growth_rate_df = intensification_df.groupby('track_id')['dvor_dt'].mean().reset_index()
growth_rate_df.rename(columns={'dvor_dt': 'mean_growth_rate'}, inplace=True)

# Converter para taxa de crescimento por segundo
tracks_df['dvor_dt'] = tracks_df['dvor_dt'] / 3600  # 1 hora = 3600 segundos

# Filtrar apenas os track_ids contidos em clusters_df
growth_rate_intense_df = growth_rate_df[growth_rate_df['track_id'].isin(clusters_df['track_id'])]
growth_rate_intense_df = growth_rate_intense_df.merge(clusters_df[['track_id', 'cluster']], on='track_id', how='left')

# Criar figura com 2x2 subplots (excluindo a taxa de crescimento médio, que será feita separadamente)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1) Contagem de sistemas por cluster
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=axes[0, 0], palette='deep')
axes[0, 0].set_title("(A) Count of Systems", fontsize=title_fontsize, fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    axes[0, 0].text(i, v + 2, str(v), ha='center', fontsize=tick_labelsize, fontweight='bold')
axes[0, 0].set_xlabel('Cluster', fontsize=ylabel_fontsize)
axes[0, 0].set_ylabel('Number of Systems', fontsize=ylabel_fontsize)
axes[0, 0].tick_params(axis='x', labelsize=tick_labelsize)
axes[0, 0].tick_params(axis='y', labelsize=tick_labelsize)

# 2) Intensidade máxima (vor42) por cluster
sns.boxplot(x=intensity_df['cluster'], y=intensity_df['vor42'], ax=axes[0, 1], palette='deep')
axes[0, 1].set_title("(B) Maximum Intensity", fontsize=title_fontsize, fontweight='bold')
axes[0, 1].set_xlabel('Cluster', fontsize=ylabel_fontsize)
axes[0, 1].set_ylabel(r'Maximum $\zeta_{850}$ ($-10^{-5}$ s$^{-1}$)', fontsize=ylabel_fontsize)
axes[0, 1].tick_params(axis='x', labelsize=tick_labelsize)
axes[0, 1].tick_params(axis='y', labelsize=tick_labelsize)

# 3) Sazonalidade dos sistemas por cluster (multi-barras)
seasonal_counts.plot(kind='bar', ax=axes[1, 0], color=season_colors)
axes[1, 0].set_title("(C) Seasonality of Systems", fontsize=title_fontsize, fontweight='bold')
axes[1, 0].set_xlabel('Cluster', fontsize=ylabel_fontsize)
axes[1, 0].set_ylabel('Frequency of Occurrence (%)', fontsize=ylabel_fontsize)
axes[1, 0].tick_params(axis='x', labelsize=tick_labelsize)
axes[1, 0].tick_params(axis='y', labelsize=tick_labelsize)
axes[1, 0].legend(title=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize, ncol=2)

# 4) Contagem de sistemas por região de gênese (multi-barras)
genesis_counts.plot(kind='bar', ax=axes[1, 1], color=[region_colors.get(region, 'gray') for region in genesis_counts.columns])
axes[1, 1].set_title("(D) Genesis Region Count", fontsize=title_fontsize, fontweight='bold')
axes[1, 1].set_xlabel('Cluster', fontsize=ylabel_fontsize)
axes[1, 1].set_ylabel('Number of Systems', fontsize=ylabel_fontsize)
axes[1, 1].tick_params(axis='x', labelsize=tick_labelsize)
axes[1, 1].tick_params(axis='y', labelsize=tick_labelsize)
axes[1, 1].legend(title=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)

# Ajustar layout e salvar figura principal
plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_analysis_panel.png", dpi=300)

# Criar figura separada para a taxa de crescimento médio durante intensificação
fig_growth, ax_growth = plt.subplots(figsize=(10, 6))
sns.boxplot(x=growth_rate_intense_df['cluster'], y=growth_rate_intense_df['mean_growth_rate'], ax=ax_growth, palette='deep')
ax_growth.set_title("Mean Growth Rate during Intensification", fontsize=title_fontsize, fontweight='bold')
ax_growth.set_xlabel('Cluster', fontsize=ylabel_fontsize)
ax_growth.set_ylabel(r'Mean Growth Rate (s$^{-2}$)', fontsize=ylabel_fontsize)
ax_growth.tick_params(axis='x', labelsize=tick_labelsize)
ax_growth.tick_params(axis='y', labelsize=tick_labelsize)

# Salvar figura da taxa de crescimento médio
plt.tight_layout()
plt.savefig(f"{output_dir}/growth_rate_intensification.png", dpi=300)