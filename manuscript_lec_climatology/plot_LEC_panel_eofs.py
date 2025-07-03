from PIL import Image
import matplotlib.pyplot as plt

path = "figures/LEC_eof"

for eof in range(1, 5):

    # Lista com os caminhos das imagens
    image_files = [
        f'{path}/LEC_EOF{eof}_incipient.png', 
        f'{path}/LEC_EOF{eof}_intensification.png', 
        f'{path}/LEC_EOF{eof}_mature.png', 
        f'{path}/LEC_EOF{eof}_decay.png'
    ]

    # Nomes das fases
    label_titles = ['(A)', '(B)', '(C)', '(D)']

    # Abrir as imagens
    images = [Image.open(img_file) for img_file in image_files]

    # Criar a figura para exibição das imagens em uma grade 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Loop para preencher a grade com as imagens
    for ax, img, label in zip(axes.flatten(), images, label_titles):
        ax.imshow(img)  # Mostrar a imagem
        ax.axis('off')  # Remover os eixos
        
        # Usando ax.text() para colocar o título no centro de cada imagem
        ax.text(0.5, 0.6, label, transform=ax.transAxes, fontsize=16, ha='center', va='center', color='black',
                fontweight='bold')

    # Ajustar o layout
    plt.tight_layout()
    plt.savefig(f'figures/panel_LEC_EOF{eof}.png')
