import pandas as pd
import matplotlib.pyplot as plt
import os

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
energetics_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv'
energetics_df = pd.read_csv(energetics_path)

# Remover NaNs do DataFrame
energetics_df = energetics_df.dropna(subset=['track_id', 'period', 'Ck']).reset_index(drop=True)

# Filtrar apenas ciclones com as fases na ordem exata: incipient, intensification, mature, decay
required_phases = ['incipient', 'intensification', 'mature', 'decay']

print("Original dataframe shape:", energetics_df.shape)

# Remover as linhas em que a fase = 'residual'
energetics_df = energetics_df[energetics_df['period'] != 'residual'].reset_index(drop=True)
print("Shape after removing 'residual' phases:", energetics_df.shape)

def has_required_phases(group):
    # Pega apenas a primeira ocorrência de cada fase, mantendo a ordem
    phases = group['period'].drop_duplicates().tolist()
    return phases == required_phases

# Encontrar track_ids válidos
valid_tracks = (
    energetics_df.groupby('track_id')
    .filter(has_required_phases)['track_id']
    .unique()
)
print("Number of valid tracks with required phases:", len(valid_tracks))

# Filtrar o DataFrame original para manter apenas esses track_ids
energetics_df = energetics_df[energetics_df['track_id'].isin(valid_tracks)]
print("Shape after filtering valid tracks:", energetics_df.shape)

# Filtrar apenas a fase 'decay'
decay_df = energetics_df[energetics_df['period'] == 'decay']
print("Shape of decay phase dataframe:", decay_df.shape)

# Obter o primeiro e último valor de Ck para cada ciclone na fase decay
first_Ck = decay_df.groupby('track_id').first()['Ck']
last_Ck = decay_df.groupby('track_id').last()['Ck']

# Organizar em um DataFrame para o boxplot
Ck_comparison = pd.DataFrame({
    'Ck_inicial': first_Ck,
    'Ck_final': last_Ck
}).dropna()
print("Shape of Ck comparison dataframe:", Ck_comparison.shape)

# Calcular mediana e intervalo interquartil (IQR)
medianas = Ck_comparison.median()
q1 = Ck_comparison.quantile(0.25)
q3 = Ck_comparison.quantile(0.75)
iqr = q3 - q1

# Scientific style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5, 6), dpi=300)

bars = ax.bar(['Initial Ck', 'Final Ck'], medianas, yerr=iqr, capsize=8, color=['#377eb8', '#e41a1c'], width=0.6)
ax.set_ylabel('Ck', fontsize=14)
ax.set_title('Median Ck at Start and End of Decay Phase\n(Error bar = IQR)', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Ajustar limites do eixo y para melhor visualização
ymin = min(medianas - iqr) * 0.95
ymax = max(medianas + iqr) * 1.05
ax.set_ylim([ymin, ymax])

plt.tight_layout()

# Save figure
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(f'{figures_dir}/ck_decay_phase_median_iqr.png', bbox_inches='tight', dpi=300)
plt.close()

