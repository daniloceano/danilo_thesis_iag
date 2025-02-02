import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Definir cores para sazonalidade
season_colors = {
    'JJA': '#5975A4',  # Azul
    'MAM': '#CC8963',  # Amarelo
    'DJF': '#B55D60',  # Vermelho
    'SON': '#5F9E6E'   # Verde
}

# Definir cores para regiões
region_colors = {
    'ARG': season_colors['JJA'],       # Azul para ARG
    'LA-PLATA': season_colors['MAM'],  # Amarelo para LA-PLATA
    'SE-BR': season_colors['SON']      # Verde para SE-BR
}

# Caminhos para os arquivos
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
track_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
pcs_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs.csv'

# Criar diretório para salvar as figuras
output_dir = 'figures/eof_intense_systems'
os.makedirs(output_dir, exist_ok=True)

# Carregar os dados
tracks = pd.read_csv(track_path)
pcs = pd.read_csv(pcs_path)

# Converter a coluna 'date' para datetime
tracks['date'] = pd.to_datetime(tracks['date'])

# Determinar os quantis da intensidade máxima (vor42)
q90 = tracks['vor42'].quantile(0.90)
q95 = tracks['vor42'].quantile(0.95)
q99 = tracks['vor42'].quantile(0.99)

# Categorizar intensidade
def categorize_intensity(value):
    if q90 <= value < q95:
        return 'q90-q95'
    elif q95 <= value < q99:
        return 'q95-q99'
    elif value >= q99:
        return 'q99+'
    return None

tracks['intensity_category'] = tracks['vor42'].apply(categorize_intensity)

# Filtrar apenas os ciclones mais intensos e manter a região de gênese e estação
tracks['season'] = tracks['date'].dt.month.map(lambda m: 'DJF' if m in [12,1,2] else 'MAM' if m in [3,4,5] else 'JJA' if m in [6,7,8] else 'SON')
extreme_tracks = tracks.dropna(subset=['intensity_category'])[['track_id', 'intensity_category', 'region', 'season', 'vor42']]

# Unir com as PCs, mantendo apenas os ciclones extremos
pcs_filtered = pcs.merge(extreme_tracks, on='track_id')

# Criar um DataFrame no formato longo para o Seaborn (PCs)
melted_pcs = pcs_filtered.melt(id_vars=['track_id', 'intensity_category'], 
                               value_vars=['PC1', 'PC2', 'PC3', 'PC4'],
                               var_name='PC', value_name='Value')

# Determinar a EOF predominante para cada ciclone com base na PC de maior valor absoluto
pcs_filtered["dominant_eof"] = pcs_filtered[['PC1', 'PC2', 'PC3', 'PC4']].abs().idxmax(axis=1)
pcs_filtered["dominant_eof"] = pcs_filtered["dominant_eof"].str.extract(r'(\d+)').astype(int)

# Criar o painel 2x2
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# **(A) Boxplot das PCs para ciclones intensos**
sns.boxplot(data=melted_pcs, x='PC', y='Value', hue='intensity_category', palette="viridis", ax=axes[0, 0])
axes[0, 0].set_xlabel("Principal Component (PC)", fontsize=16)
axes[0, 0].set_ylabel("PC Value", fontsize=16)
axes[0, 0].set_title("(A) PC Distribution for Extreme Cyclones", fontsize=18, fontweight="bold")
axes[0, 0].legend(title="Intensity Category", fontsize=12)

# **(B) Boxplot da intensidade máxima dos sistemas por EOF**
sns.boxplot(data=pcs_filtered, x="dominant_eof", y="vor42", palette="muted", ax=axes[0, 1])
axes[0, 1].set_xlabel("EOF", fontsize=16)
axes[0, 1].set_ylabel("Maximum Intensity", fontsize=16)
axes[0, 1].set_title("(B) Max Intensity per EOF", fontsize=18, fontweight="bold")

# **(C) Contagem de sistemas por EOF e região de gênese**
ax = sns.countplot(data=pcs_filtered, x="dominant_eof", hue="region", palette=region_colors, ax=axes[1, 0])

# Adicionar rótulos numéricos acima das barras
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=14, fontweight='bold')

axes[1, 0].set_xlabel("EOF", fontsize=16)
axes[1, 0].set_ylabel("Number of Cyclones", fontsize=16)
axes[1, 0].set_title("(C) Cyclone Count by EOF and Genesis Region", fontsize=18, fontweight="bold")
axes[1, 0].legend(title="Region", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# **(D) Contagem de sistemas por EOF e estação do ano**
ax = sns.countplot(data=pcs_filtered, x="dominant_eof", hue="season", palette=season_colors, ax=axes[1, 1])

# Adicionar rótulos numéricos acima das barras
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=14, fontweight='bold')

axes[1, 1].set_xlabel("EOF", fontsize=16)
axes[1, 1].set_ylabel("Number of Cyclones", fontsize=16)
axes[1, 1].set_title("(D) Cyclone Count by EOF and Season", fontsize=18, fontweight="bold")
axes[1, 1].legend(title="Season", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar layout e salvar a figura
plt.tight_layout()
output_file_panel = os.path.join(output_dir, 'panel_2x2_eof_intense_systems.png')
plt.savefig(output_file_panel, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figura salva em: {output_file_panel}")
