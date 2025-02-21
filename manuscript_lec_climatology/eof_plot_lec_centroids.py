import pandas as pd
from plot_LEC_std import plot_lorenzcycletoolkit_with_std, read_life_cycles

if __name__ == "__main__":

    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['BA', 'BK'],
        'Pressure Work Terms': ['BΦ'],
        'Generation/Residual Terms': ['G', 'R'],
        'Budget Terms': ['∂']
    }

    terms_prefix = list(groups.keys())

    # Caminhos dos arquivos
    PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
    figures_directory = "./figures/eof_clusters_intense/"
    eofs_path = f'{PATH}/csv_eofs_energetics_with_track/Total/eofs.csv'
    centroids_path = f'{figures_directory}/cluster_centroids.csv'
    base_path = f'{PATH}/csv_database_energy_by_periods'
    pcs_with_clusters_path = f'{figures_directory}/pcs_with_clusters.csv'

    # Carregar os dados dos tracks
    systems_energetics = read_life_cycles(base_path)

    # Carregar os dados das PCs
    pcs_with_clusters = pd.read_csv(pcs_with_clusters_path)

    # Concatenate all systems' dataframes while retaining the system id and phase
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
    all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)
    
    # Convert relevant columns to numeric, forcing errors to NaN
    relevant_columns = ['system_id'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    all_data[relevant_columns] = all_data[relevant_columns].apply(pd.to_numeric, errors='coerce')
    
    # Compute mean across all phases for each system
    mean_data = all_data.drop('Phase', axis=1).groupby('system_id').mean().reset_index().mean().drop('system_id')

    # Retirar ((finite diff.) dos nomes das colunas
    mean_data = mean_data.rename(lambda x: x.replace(" (finite diff.)", ""))

    # Carregar os dados
    eofs = pd.read_csv(eofs_path)
    centroids = pd.read_csv(centroids_path)

    # Verificar dimensões
    print("Dimensões de centroids:", centroids.shape)  # Deve ser (5, 8)
    print("Dimensões de eofs:", eofs.shape)  # Deve ser (8, 24)

    # Remover colunas não numéricas (se houver)
    eofs = eofs.select_dtypes(include=['number'])
    centroids = centroids.select_dtypes(include=['number'])

    # Verificar novamente após limpeza
    print("Dimensões corrigidas de centroids:", centroids.shape)
    print("Dimensões corrigidas de eofs:", eofs.shape)

    # Garantir que 'eofs' tem (8, 24)
    if eofs.shape[0] == 8 and eofs.shape[1] == 24:
        print("EOFs está corretamente formatado!")

    # Multiplicação das matrizes
    energetics = centroids.values @ eofs.values  # Dimensão resultante: (5, 24)
    print("Multiplicação bem-sucedida!")
    
    # Criar DataFrame
    energetics_df = pd.DataFrame(energetics, columns=eofs.columns)

    # Criar um índice nomeado para os clusters
    energetics_df.index = [f'Cluster {i+1}' for i in range(len(energetics_df))]

    # Somar os valores médios aos centroids
    energetics_df_sum = energetics_df + mean_data

    # Criar df com energética separada por cluster
    all_data_renamed = all_data.rename(columns={'system_id': 'track_id'})
    all_data_with_cluster = pd.merge(all_data_renamed, pcs_with_clusters[['track_id', 'cluster']], on='track_id')
    all_data_with_cluster['cluster'] += 1
    all_data_with_cluster = all_data_with_cluster[all_data_with_cluster['track_id'].isin(pcs_with_clusters['track_id'])]
    all_data_with_cluster = all_data_with_cluster.drop('Phase', axis=1).groupby('track_id').mean()
    energetics_df_std = all_data_with_cluster.groupby('cluster').std()
    energetics_df_std = energetics_df_std.rename(columns=lambda x: x.replace(" (finite diff.)", ""))
    energetics_df_std.index = [f'Cluster {i+1}' for i in range(len(energetics_df_std))]

    # Plot Lorenz cycle
    plot_lorenzcycletoolkit_with_std(energetics_df_sum, energetics_df_std, figures_directory)


