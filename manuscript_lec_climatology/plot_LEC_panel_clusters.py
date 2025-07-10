from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path = "figures/eof_clusters_intense/LEC_std"
image_files = [
    'figures/LEC_std/LEC_total.png',
    f'{path}/LEC_Cluster 1.png',
    f'{path}/LEC_Cluster 2.png',
    f'{path}/LEC_Cluster 3.png',
    f'{path}/LEC_Cluster 4.png',
]

# Nomes das fases
label_titles = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']

# Abrir as imagens
images = [Image.open(img_file) for img_file in image_files]

# Criar a figura
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loop para preencher a grade com as imagens
for ax, img, label in zip(axes.flatten(), images, label_titles):
    print(f'Plotting {label}...')  # Log para saber qual imagem está sendo processada
    print('Axes:', ax)
    # plot apenas se a imagem não for None
    if img is not None:
        ax.imshow(img)  # Mostrar a imagem
        ax.axis('off')  # Remover os eixos

        # Se for a primeira imagem, usar caixa branca em volta da label para destacar
        if label == '(A)':
            bbox = patches.Rectangle((0.4, 0.4), 0.2, 0.2, linewidth=2, edgecolor='white', facecolor='white',
                                    transform=ax.transAxes)
            ax.add_patch(bbox)
        # Usando ax.text() para colocar o título no centro de cada imagem
        ax.text(0.5, 0.55, label, transform=ax.transAxes, fontsize=16, ha='center', va='center', color='black',
                fontweight='bold')
        if label == '(A)':
            ax.text(0.5, 0.47, 'All Systems\nMean ± Std', transform=ax.transAxes, fontsize=8, ha='center', va='center', color='black',
                fontweight='bold')
    else:
        fig.delaxes(ax)  # Remove o subplot se não houver imagem

# Remover os eixos das imagens que não foram usadas
for ax in axes.flatten()[len(images):]:
    ax.axis('off')

# Ajustar o layout
plt.tight_layout()
plt.savefig('figures/eof_clusters_intense/panel_LEC_clusters.png')
