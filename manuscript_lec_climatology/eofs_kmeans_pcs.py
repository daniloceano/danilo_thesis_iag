import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Caminho dos arquivos
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
pcs_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs.csv'
output_dir = 'figures/eof_clusters_intense'

# Criar diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Carregar os dados dos tracks
track_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
tracks = pd.read_csv(track_path)

# Extrair sistemas intensos
q99 = tracks['vor42'].quantile(0.99)
tracks_intense = tracks[tracks['vor42'] >= q99]

# Carregar os dados das PCs
pcs_df = pd.read_csv(pcs_path)

# Filtrar apenas os sistemas intensos
pcs_df = pcs_df[pcs_df['track_id'].isin(tracks_intense['track_id'])]

# Remover a coluna track_id, pois não deve ser usada na clusterização
data = pcs_df.drop(columns=['track_id'])

# Normalizar os dados para evitar influência de escalas diferentes
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determinar o número ótimo de clusters pelo método Elbow
fig, ax = plt.subplots(figsize=(8, 5))
visualizer = KElbowVisualizer(KMeans(), k=(2, 10), metric='distortion', timings=False, ax=ax)
visualizer.fit(data_scaled)  # Treinar modelo KMeans
visualizer.show(outpath=f"{output_dir}/elbow_method.png")

# Número ótimo de clusters
optimal_k = visualizer.elbow_value_
print(f"Número ótimo de clusters: {optimal_k}")

# Aplicar K-means com o número ótimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
pcs_df['cluster'] = kmeans.fit_predict(data_scaled)

# Salvar os centróides em um arquivo CSV
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=data.columns)
centroids.to_csv(f"{output_dir}/cluster_centroids.csv", index=False)

# Salvar os resultados da clusterização
pcs_df.to_csv(f"{output_dir}/pcs_with_clusters.csv", index=False)

# Exibir resultados
print("Clusterização concluída! Os arquivos foram salvos em:", output_dir)
