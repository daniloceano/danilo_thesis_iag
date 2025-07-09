import matplotlib.pyplot as plt
import pandas as pd

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
energetics_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv'
energetics_df = pd.read_csv(energetics_path)

# Remover NaNs do DataFrame
energetics_df = energetics_df.dropna(subset=['track_id', 'period', 'Ck']).reset_index(drop=True)

# Calcular a intensidade máxima de cada ciclone
max_intensity = energetics_df.groupby('track_id')['vor42'].max()

# Calcular o valor do quantil 90
q90 = max_intensity.quantile(0.90)

# Contar quantos ciclones estão no quantil 90
n_q90 = (max_intensity >= q90).sum()

# Plotar histograma (PDF) com bins de 1% do range
bins = int((max_intensity.max() - max_intensity.min()) / (0.01 * (max_intensity.max() - max_intensity.min())))
plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(max_intensity, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(q90, color='red', linestyle='--', label=f'Quantil 90%\n({n_q90} ciclones)')
plt.xlabel('Intensidade máxima')
plt.ylabel('PDF')
plt.title('PDF da intensidade máxima dos ciclones')
plt.legend()
plt.tight_layout()
plt.show()
