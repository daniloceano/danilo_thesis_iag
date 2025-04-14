import pandas as pd
import requests
from pathlib import Path

# === Carrega o dataframe ===
file = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
df = pd.read_csv(file, parse_dates=['date'])

# === Define critérios de filtragem ===
target_sequence = ['incipient', 'intensification', 'mature', 'decay']

def has_ordered_phases(group):
    phases = group['period'].tolist()
    filtered = [p for i, p in enumerate(phases) if i == 0 or p != phases[i-1]]
    return all(p in filtered for p in target_sequence) and \
           filtered.index('incipient') < filtered.index('intensification') < filtered.index('mature') < filtered.index('decay')

def within_atlantic_south(group):
    return (
        group['lon vor'].between(-70, -10).all() and
        group['lat vor'].between(-60, 0).all()
    )

# === Aplica filtros ===
df_valid = df.groupby('track_id').filter(has_ordered_phases)
df_valid = df_valid[df_valid['date'] >= '2002-07-04']  # garante datas com MODIS/VIIRS
df_valid = df_valid.groupby('track_id').filter(within_atlantic_south)

# === Calcula duração por ciclone ===
duracoes = df_valid.groupby('track_id').agg(
    duracao_horas=('date', lambda x: (x.max() - x.min()).total_seconds() / 3600)
)
# Filtra para pelo menos 5 dias
track_ids_validos = duracoes[duracoes['duracao_horas'] >= 120].index.tolist()

# === Sorteia 5 ciclones aleatórios ===
import random
track_ids_amostrados = random.sample(track_ids_validos, min(5, len(track_ids_validos)))
print("Ciclones selecionados:", track_ids_amostrados)

# === Parâmetros da API ===
camadas = [
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor"
]

# === Diretório base de saída ===
base_output = Path("imagens_sat")
base_output.mkdir(exist_ok=True)

# === Loop por ciclone ===
for track_id in track_ids_amostrados:
    df_ciclone = df_valid[df_valid['track_id'] == track_id]

    datas = df_ciclone[['date']].drop_duplicates().sort_values(by='date')
    datas = datas[datas['date'] >= '2002-07-04']  # reforça segurança

    # Define bbox com margem e proporção 1:1
    lon_min = df_ciclone['lon vor'].min() - 2
    lon_max = df_ciclone['lon vor'].max() + 2
    lat_min = df_ciclone['lat vor'].min() - 2
    lat_max = df_ciclone['lat vor'].max() + 2
    bbox = [lat_min, lon_min, lat_max, lon_max]

    print(f"\nTrack ID: {track_id} — BBOX: {bbox}")

    # Pasta específica para esse ciclone
    output_dir = base_output / f"{track_id}"
    output_dir.mkdir(exist_ok=True)

    for date in datas['date']:
        date_iso = pd.to_datetime(date).strftime('%Y-%m-%d')
        filename = output_dir / f"{track_id}_{date_iso}.jpg"
        
        if filename.exists():
            print(f"{filename.name} já existe, pulando.")
            continue

        success = False
        for camada in camadas:
            url = (
                f"https://wvs.earthdata.nasa.gov/api/v1/snapshot"
                f"?REQUEST=GetSnapshot&TIME={date_iso}"
                f"&BBOX={','.join(map(str, bbox))}"
                f"&CRS=EPSG:4326"
                f"&LAYERS={camada}"
                f"&FORMAT=image/jpeg&WIDTH=1024&HEIGHT=1024"
            )

            print(f"Tentando {camada} para {date_iso}...")
            r = requests.get(url)
            if r.status_code == 200 and r.headers['Content-Type'].startswith('image'):
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f"✓ Imagem salva: {filename.name}")
                success = True
                break
            else:
                print(f"✗ Falha com {camada}: {r.status_code} ({r.headers.get('Content-Type')})")

        if not success:
            print(f"⚠️ Nenhuma imagem disponível para {date_iso}")
