# era5_ciclone.py
import cdsapi
import pandas as pd
from datetime import datetime

# ---- 1. Ler track ----------------------------------------------------------
csv_path = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/LEC_Results_test_Dias-Pinto/20080547_ERA5_sliced_track/20080547_ERA5_sliced_track_trackfile'         # substitua caso o nome mude
df = pd.read_csv(csv_path, sep=';')

# ---- 2. Delimitar domínio (+8° de folga) -----------------------------------
min_lat = df['min_lat'].min() - 8
max_lat = df['max_lat'].max() + 8
min_lon = df['min_lon'].min() - 8
max_lon = df['max_lon'].max() + 8

#  Ajuste para que os valores não ultrapassem limites válidos
min_lat = max(min_lat, -90)
max_lat = min(max_lat,  90)
min_lon = max(min_lon, -180)
max_lon = min(max_lon, 180)

area = [max_lat, min_lon, min_lat, max_lon]   # [N, W, S, E]

# ---- 3. Preparar lista de datas / horas ------------------------------------
start = datetime.strptime(df['time'].iloc[0][:10], "%Y-%m-%d")  # 2008-06-29
end   = datetime.strptime(df['time'].iloc[-1][:10], "%Y-%m-%d") # 2008-07-03
date_str = f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"

hours = ["00:00","03:00","06:00","09:00","12:00","15:00","18:00","21:00"]

# ---- 4. Pedido ao CDS ------------------------------------------------------
pressure_levs = [
    "1000","925","850","700","500","400","300",
    "250","200","150","100","70","50","30","20","10"
]

variables = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature"
]

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": variables,
        "pressure_level": pressure_levs,
        "date": date_str,
        "time": hours,
        "area": area,        # [N, W, S, E]
        "format": "netcdf"   # resultará em era5_ciclone.nc
    },
    "20080547_ERA5.nc"
)

print("Download concluído → 20080547_ERA5.nc")
