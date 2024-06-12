import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_species_count(region, season):
    df = pd.read_csv(f'../results_chapter_4/count_systems/{season}_count_of_systems_{region}.csv')

    df_grouped = df.groupby('Type of System', as_index=False).sum()
    total_count = df_grouped['Total Count'].sum()
    df_grouped['Percentage'] = (df_grouped['Total Count'] / total_count) * 100

    return df_grouped

def apply_label_mapping(df, label_mapping):
    df['Type of System'] = df['Type of System'].astype(str)
    df['Type of System'] = df['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))
    return df

def plot_barplot(df_filtered, region, season, ax, suffix, color_mapping):
    df_filtered['Color'] = df_filtered['Type of System'].map(color_mapping)
    df_filtered = df_filtered.sort_values('Percentage', ascending=False)
    
    # Set the order of the y-axis categories
    y_order = df_filtered['Type of System'].values
    
    sns.barplot(x='Percentage', y='Type of System', data=df_filtered, hue='Type of System', orient='h', errorbar=None, 
                palette=df_filtered.set_index('Type of System')['Color'].to_dict(), edgecolor='grey', ax=ax, 
                legend=False, order=y_order)
    
    for bar, (_, row) in zip(ax.patches, df_filtered.iterrows()):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, 
                f"{row['Total Count']} ({row['Percentage']:.2f}%)", 
                va='center', color='black', fontsize=16)

    ax.set_xlabel('Percentage', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_ylabel(None)
    ax.set_yticklabels(df_filtered['Type of System'])

    ax.title.set_text(f'{region} - {season}')
    ax.title.set_fontsize(20)

    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

# Ensure you call this updated function within plot_multi_panel_barplots


def plot_multi_panel_barplots(regions, seasons, output_directory, suffix):
    fig, axes = plt.subplots(len(regions), len(seasons), figsize=(24, 18), sharex='col', sharey='row')

    life_cycles_filtered = pd.read_csv(f'../results_chapter_4/total_count_of_systems_filtered.csv')
    
    label_mapping = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
        'residual': 'R'
    }
    
    # Apply the label mapping to the life cycles
    life_cycles_filtered = apply_label_mapping(life_cycles_filtered, label_mapping)
    
    # Generate a unique color for each life cycle configuration
    unique_life_cycles = life_cycles_filtered['Type of System'].unique()
    palette = sns.color_palette("pastel", len(unique_life_cycles))
    color_mapping = dict(zip(unique_life_cycles, palette))

    for i, region in enumerate(regions):
        for j, season in enumerate(seasons):
            df = get_species_count(region, season)
            df = apply_label_mapping(df, label_mapping)
            df_filtered = df[df['Type of System'].isin(life_cycles_filtered['Type of System'])]
            df_filtered = df_filtered.sort_values(by='Total Count', ascending=False)

            plot_barplot(df_filtered, region, season, axes[i, j], suffix, color_mapping)

    # Create a custom legend
    handles = [plt.Rectangle((0,0),1,1, color=color_mapping[lc]) for lc in unique_life_cycles]
    plt.figlegend(handles, unique_life_cycles, bbox_to_anchor=(0.5, 0.1), loc='upper center', ncol=4, prop={'size': 20})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.1)

    combined_output_file = os.path.join(output_directory, f'multi_panel_barplots_{suffix}.png')
    plt.savefig(combined_output_file)
    print(f'{combined_output_file} saved.')

output_directory = '../figures_chapter_4/'
os.makedirs(output_directory, exist_ok=True)

regions = ['ARG', 'LA-PLATA', 'SE-BR']
seasons = ['DJF', 'MAM', 'JJA', 'SON']

plot_multi_panel_barplots(regions, seasons, output_directory, 'filtered')
