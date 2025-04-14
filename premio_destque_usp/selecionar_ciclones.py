import pandas as pd

# Carrega o arquivo
file = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
df = pd.read_csv(file, parse_dates=['date'])

# Fases obrigatórias
target_sequence = ['incipient', 'intensification', 'mature', 'decay']

def has_ordered_phases(group):
    phases = group['period'].tolist()
    filtered = [phase for i, phase in enumerate(phases) if phase != phases[i-1] or i == 0]
    return all(p in filtered for p in target_sequence) and \
           filtered.index('incipient') < filtered.index('intensification') < filtered.index('mature') < filtered.index('decay')

# Aplica filtro por fases
df_valid = df.groupby('track_id').filter(has_ordered_phases)

# Filtra por data mínima
df_valid = df_valid[df_valid['date'] >= '2017-01-01']

# Filtra para ciclones inteiramente dentro do Atlântico Sul
def within_atlantic_south(group):
    return (
        group['lon vor'].between(-70, -10).all() and
        group['lat vor'].between(-60, 0).all()
    )

df_valid = df_valid.groupby('track_id').filter(within_atlantic_south)

# Calcula duração e bbox
duracoes = df_valid.groupby('track_id').agg(
    data_inicio=('date', 'min'),
    data_fim=('date', 'max'),
    duracao_horas=('date', lambda x: (x.max() - x.min()).total_seconds() / 3600),
    lat_min=('lat vor', 'min'),
    lat_max=('lat vor', 'max'),
    lon_min=('lon vor', 'min'),
    lon_max=('lon vor', 'max')
).sort_values(by='duracao_horas', ascending=False)

# Seleciona os 10 mais duradouros
top10 = duracoes.head(10)

# Imprime resultados
for idx, row in top10.iterrows():
    print(f"Track ID: {idx}")
    print(f"  Início: {row['data_inicio'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Fim:    {row['data_fim'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Duração: {row['duracao_horas']:.1f} horas")
    print(f"  BBOX: [lat: {row['lat_min']:.2f} to {row['lat_max']:.2f}, lon: {row['lon_min']:.2f} to {row['lon_max']:.2f}]")
    print()
