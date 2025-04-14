import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Caminhos
csv_file = '20202365.csv'
image_folder = Path('imagens_visivel')
output_folder = Path('imagens_rotuladas')
output_folder.mkdir(exist_ok=True)

# Carrega CSV com fases
df = pd.read_csv(csv_file, parse_dates=['date'])

# Preenche fases ausentes
df['period'] = df['period'].fillna(method='bfill').fillna(method='ffill')

# Mapeia timestamp -> fase
mapa_fases = {
    d.strftime('%Y%m%d%H%M'): fase for d, fase in zip(df['date'], df['period'])
}

# Fonte
try:
    font_path = "/System/Library/Fonts/HelveticaNeue.ttc"
    font = ImageFont.truetype(font_path, size=30)
    print("✓ Fonte HelveticaNeue carregada com sucesso")
except Exception as e:
    font = ImageFont.load_default()
    print("⚠️ Fonte padrão usada:", e)

# Processa imagens
for img_path in image_folder.glob('*.png'):
    timestamp = img_path.stem.split('_')[1]  # formato: YYYYMMDDHHMM

    fase = mapa_fases.get(timestamp)

    # Se não houver fase direta, tenta bfill e depois ffill
    if not fase:
        # Converte para datetime
        ts_dt = datetime.strptime(timestamp, "%Y%m%d%H%M")

        # Tenta encontrar próxima data disponível
        datas_ordenadas = sorted(mapa_fases.keys())
        posteriores = [k for k in datas_ordenadas if k > timestamp]
        anteriores = [k for k in datas_ordenadas if k < timestamp]

        if posteriores:
            fase = mapa_fases[posteriores[0]]
        elif anteriores:
            fase = mapa_fases[anteriores[-1]]

    # Se ainda não tiver fase, pula (caso raro)
    if not fase:
        print(f"⚠️ Nenhuma fase atribuída para {timestamp}, pulando.")
        continue

    # Abre imagem
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Texto e posição
    text = fase.upper()
    x, y = 70, 80

    # Calcula tamanho do texto e caixa
    text_bbox = draw.textbbox((x, y), text, font=font)
    padding = 10
    box_coords = [
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    ]

    # Caixa branca
    draw.rectangle(box_coords, fill='white')

    # Texto preto sobre a caixa
    draw.text((x, y), text, font=font, fill='black')

    # Salva imagem modificada
    output_file = output_folder / img_path.name
    img.save(output_file)
    print(f"✓ Fase '{fase}' adicionada em {output_file.name}")


# Caminho das imagens rotuladas
image_folder = Path("imagens_rotuladas")

# Ordena os arquivos por nome
image_files = sorted(image_folder.glob("*.png"))

# Abre todas as imagens
frames = [Image.open(img_path).convert("RGB") for img_path in image_files]

# Salva como GIF animado
output_gif = "20202365.gif"
frames[0].save(
    output_gif,
    format='GIF',
    save_all=True,
    append_images=frames[1:],
    duration=100,  # duração de cada frame em ms
    loop=0  # loop infinito
)

print(f"✓ GIF salvo como {output_gif}")
