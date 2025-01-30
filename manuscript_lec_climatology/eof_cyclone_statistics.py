import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'  # Dezembro, Janeiro, Fevereiro
    elif month in [3, 4, 5]:
        return 'MAM'  # Março, Abril, Maio
    elif month in [6, 7, 8]:
        return 'JJA'  # Junho, Julho, Agosto
    elif month in [9, 10, 11]:
        return 'SON'  # Setembro, Outubro, Novembro
    
season_colors = {
    'JJA': '#5975A4',  # Azul
    'MAM': '#CC8963',  # Amarelo
    'DJF': '#B55D60',  # Vermelho
    'SON': '#5F9E6E'   # Verde
}

# Caminhos para os arquivos
track_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
pcs_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof.csv'

# Criar diretório para salvar as figuras
output_dir = 'figures/eof_statistics'
os.makedirs(output_dir, exist_ok=True)

# Carregar os dados
tracks = pd.read_csv(track_path)
pcs = pd.read_csv(pcs_path)

# Converter a coluna 'date' para datetime
tracks['date'] = pd.to_datetime(tracks['date'])

# Unir os dados com base no track_id
merged_data = tracks.merge(pcs[['track_id', 'dominant_eof']], on='track_id')

# Restringir análise às primeiras 4 EOFs
merged_data = merged_data[merged_data['dominant_eof'].isin([1, 2, 3, 4])]

# Calcular estatísticas por ciclone
cyclone_stats = merged_data.groupby(['track_id', 'dominant_eof']).agg(
    max_intensity=('vor42', lambda x: x.max()), 
    mean_intensity=('vor42', lambda x: x.mean()), 
    duration=('date', lambda x: (x.max() - x.min()).total_seconds() / (3600 * 24))  # Duração em dias
).reset_index()

# Adicionar a região de ciclogênese no cálculo das estatísticas
cyclone_stats_by_region = merged_data.groupby(['track_id', 'dominant_eof', 'region']).agg(
    max_intensity=('vor42', lambda x: x.max()), 
    mean_intensity=('vor42', lambda x: x.mean()), 
    duration=('date', lambda x: (x.max() - x.min()).total_seconds() / (3600 * 24))  # Duração em dias
).reset_index()

# Calcular o número de ciclones por EOF
cyclone_count = cyclone_stats.groupby('dominant_eof').size().reset_index(name='cyclone_count')

# Determinar a gênese: selecionar a primeira entrada de cada ciclone
genesis_data = merged_data.groupby('track_id').first().reset_index()

# Contar sistemas por região de gênese e EOF
region_counts = genesis_data.groupby(['dominant_eof', 'region']).size().reset_index(name='count')

# Calcular proporção por EOF
region_counts['proportion'] = region_counts.groupby('dominant_eof')['count'].transform(lambda x: x / x.sum())

# Converter proporções para porcentagens
region_counts['proportion'] *= 100

# Pivotar os dados para formato adequado para gráfico de barras empilhadas
proportion_pivot = region_counts.pivot(index='dominant_eof', columns='region', values='proportion').fillna(0)

# ** Análise de sazonalidade **
# Adicionar coluna com a estação do ano
merged_data['month'] = merged_data['date'].dt.month
merged_data['season'] = merged_data['month'].apply(get_season)

# Contar ocorrências por EOF e estação
# Calcular a frequência de ocorrência em porcentagem por estação e EOF
season_order = ['DJF', 'MAM', 'JJA', 'SON']
seasonal_counts = merged_data.groupby(['dominant_eof', 'season']).size().reset_index(name='count')
seasonal_counts['season'] = pd.Categorical(seasonal_counts['season'], categories=season_order, ordered=True)
total_counts_per_eof = seasonal_counts.groupby('dominant_eof')['count'].sum().reset_index(name='total_count')
seasonal_counts = seasonal_counts.merge(total_counts_per_eof, on='dominant_eof')
seasonal_counts['frequency'] = (seasonal_counts['count'] / seasonal_counts['total_count']) * 100

# Pivotar para formato de gráfico
seasonal_pivot = seasonal_counts.pivot(index='dominant_eof', columns='season', values='count').fillna(0)

# Configurar estilo para publicação científica
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Paletas de cores para gráficos
palette = sns.color_palette("deep", n_colors=4)

# Boxplot da intensidade máxima por EOF e região
plt.figure(figsize=(14, 8))
sns.boxplot(data=cyclone_stats_by_region, x='dominant_eof', y='max_intensity', hue='region', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel(r'Maximum $\zeta_{850}$ ($-10^{-5}$ s$^{-1}$)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Region', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_max_intensity_per_eof_and_region.png'), dpi=300)

# Boxplot da intensidade média por EOF e região
plt.figure(figsize=(14, 8))
sns.boxplot(data=cyclone_stats_by_region, x='dominant_eof', y='mean_intensity', hue='region', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel(r'Mean $\zeta_{850}$ ($-10^{-5}$ s$^{-1}$)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Region', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_mean_intensity_per_eof_and_region.png'), dpi=300)

# Boxplot da duração por EOF e região
plt.figure(figsize=(14, 8))
sns.boxplot(data=cyclone_stats_by_region, x='dominant_eof', y='duration', hue='region', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Duration (days)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Region', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_duration_per_eof_and_region.png'), dpi=300)

# Gráfico de barras do número de ciclones por EOF
plt.figure(figsize=(10, 6))
sns.barplot(data=cyclone_count, x='dominant_eof', y='cyclone_count', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Number of cyclones', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'barplot_cyclone_count_per_eof.png'), dpi=300)

# Gráfico de barras da proporção por região de gênese e EOF
proportion_pivot.plot(kind='bar', stacked=False, figsize=(12, 8), color=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Proportion of Systems (%)', fontsize=16)  # Atualizar o rótulo do eixo Y
plt.legend(fontsize=12)
plt.xticks(rotation=0, fontsize=14)  # Ajustar rotação do eixo X
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stacked_bar_genesis_proportion_per_eof.png'), dpi=300)


# Definir tamanhos para os rótulos
ylabel_fontsize = 18
title_fontsize = 20
tick_labelsize = 16
legend_fontsize = 14

# Criar figura com painel 2x2
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Gráfico 1: Proporção por região de gênese e EOF (com Seaborn)
proportion_melted = proportion_pivot.reset_index().melt(id_vars='dominant_eof', var_name='region', value_name='proportion')

sns.barplot(
    data=proportion_melted,
    x='dominant_eof',
    y='proportion',
    hue='region',
    palette=palette,
    ax=axes[0, 0]
)
axes[0, 0].set_ylabel('Proportion of Systems (%)', fontsize=ylabel_fontsize)
axes[0, 0].tick_params(axis='x', labelsize=tick_labelsize)
axes[0, 0].tick_params(axis='y', labelsize=tick_labelsize)
axes[0, 0].legend(title=None, fontsize=legend_fontsize, bbox_to_anchor=(0.35, 1.0), ncol=3)
axes[0, 0].set_title('(A) Genesis Proportion by Region and EOF', fontsize=title_fontsize, fontweight='bold')
axes[0, 0].set_xlabel('')  # Remover label do eixo X

# Gráfico 2: Sazonalidade
sns.barplot(
    data=seasonal_counts,
    x='dominant_eof',
    y='frequency',
    hue='season',
    palette=season_colors,
    hue_order=season_order,
    ax=axes[0, 1]
)
axes[0, 1].set_ylabel('Frequency of Occurrence (%)', fontsize=ylabel_fontsize)
axes[0, 1].tick_params(axis='x', labelsize=tick_labelsize)
axes[0, 1].tick_params(axis='y', labelsize=tick_labelsize)
axes[0, 1].legend(title=None, fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1.0))
axes[0, 1].set_title('(B) Seasonal Occurrences per EOF', fontsize=title_fontsize, fontweight='bold')
axes[0, 1].set_xlabel('')  # Remover label do eixo X

# Gráfico 3: Intensidade máxima por EOF (sem região)
sns.boxplot(data=cyclone_stats, x='dominant_eof', y='max_intensity', palette=palette, ax=axes[1, 0])
axes[1, 0].set_ylabel(r'Maximum $\zeta_{850}$ ($-10^{-5}$ s$^{-1}$)', fontsize=ylabel_fontsize)
axes[1, 0].tick_params(axis='x', labelsize=tick_labelsize)
axes[1, 0].tick_params(axis='y', labelsize=tick_labelsize)
axes[1, 0].set_title('(C) Maximum Intensity per EOF', fontsize=title_fontsize, fontweight='bold')
axes[1, 0].set_xlabel('')  # Remover label do eixo X

# Gráfico 4: Duração média por EOF (sem região)
sns.boxplot(data=cyclone_stats, x='dominant_eof', y='duration', palette=palette, ax=axes[1, 1])
axes[1, 1].set_ylabel('Duration (days)', fontsize=ylabel_fontsize)
axes[1, 1].tick_params(axis='x', labelsize=tick_labelsize)
axes[1, 1].tick_params(axis='y', labelsize=tick_labelsize)
axes[1, 1].set_title('(D) Duration per EOF', fontsize=title_fontsize, fontweight='bold')
axes[1, 1].set_xlabel('')  # Remover label do eixo X

# Ajustar layout e salvar a figura
plt.tight_layout()
output_path = os.path.join(output_dir, 'panel_2x2_eof_statistics.png')
plt.savefig(output_path, dpi=300)
plt.close()
print(f'Panel saved to {output_path}')
