import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== CONFIG =====================
BASE_DIR = 'tests_LEC_domain_size'
INDIVIDUAL_DIR = os.path.join(BASE_DIR, 'comparing_terms/individual')
PANEL_DIR = os.path.join(BASE_DIR, 'comparing_terms/panels')
SIZES_COLORS = [
    ('20x20', '#FFD700'),  # yellow
    ('15x15', '#FFA500'),  # orange
    ('10x10', '#FF0000'),  # red
    ('5x5',  '#800080'),   # purple
]
UNIT_MAP = {'Az': 'J s⁻¹', 'Ae': 'J s⁻¹', 'Kz': 'J s⁻¹', 'Ke': 'J s⁻¹'}
PANEL_GROUPS = {
    'Energy Terms': ['Az', 'Ae', 'Kz', 'Ke'],
    'Conversion Terms': ['Cz', 'Ca', 'Ck', 'Ce'],
    'Boundary Terms': ['BAz', 'BAe', 'BKz', 'BKe'],
    'Pressure Work Terms': ['BΦZ', 'BΦE'],
    'Generation_Residual_Terms': ['Gz', 'Ge', 'RGz', 'RKz', 'RGe', 'RKe'],
    'Budget Terms': ['Az_budget', 'Ae_budget', 'Kz_budget', 'Ke_budget'],
}

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

# ===================== FUNÇÕES =====================
def read_all_dataframes(sizes_colors, base_dir):
    """Lê todos os dataframes dos diferentes tamanhos de domínio."""
    dfs = {}
    for size, _ in sizes_colors:
        folder = os.path.join(base_dir, f'20080547_ERA5_track_{size}')
        csv_path = os.path.join(folder, '20080547_ERA5_track_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            time_col = 'time' if 'time' in df.columns else df.columns[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                pass
            dfs[size] = (df, time_col)
        else:
            print(f'File not found: {csv_path}')
    return dfs

def extract_terms_and_labels(example_df, time_col):
    """Extrai os termos e renomeia os termos delta para *_budget."""
    terms = []
    term_labels = {}
    for col in example_df.columns:
        if col == time_col:
            continue
        if col.startswith('∂') and 'finite diff.' in col:
            main = col[1:col.index('/')]
            new_name = f'{main}_budget'
            terms.append(col)
            term_labels[col] = new_name
        else:
            terms.append(col)
            term_labels[col] = col
    return terms, term_labels

def get_unit(term, term_labels):
    """Retorna a unidade do termo."""
    base = term_labels.get(term, term)
    if base.endswith('_budget'):
        base = base.replace('_budget', '')
    return UNIT_MAP.get(base, 'W²')

def plot_individual_terms(terms, term_labels, dfs, sizes_colors, output_dir):
    """Plota gráficos individuais para cada termo."""
    for term in terms:
        plt.figure(figsize=(10, 6))
        plotted = False
        for (size, color) in sizes_colors:
            if size in dfs and term in dfs[size][0].columns:
                df, time_col = dfs[size]
                plt.plot(df[time_col], df[term], label=size, color=color)
                plotted = True
        if plotted:
            label = term_labels[term]
            plt.xlabel('Date')
            plt.ylabel(f'{label} ({get_unit(term, term_labels)})')
            plt.title(f'Time series of {label} for different domain sizes')
            base_label = label.replace('_budget', '')
            # Limitar eixo y a 0 apenas para termos de energia
            if base_label in UNIT_MAP:
                plt.ylim(bottom=0)
            else:
                plt.axhline(0, color='black', linestyle='-', linewidth=1)
            plt.legend(title='Domain size')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %HZ'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            all_times = []
            for (size, _) in sizes_colors:
                if size in dfs and term in dfs[size][0].columns:
                    df, time_col = dfs[size]
                    all_times.extend(df[time_col])
            if all_times:
                ax.set_xlim(min(all_times), max(all_times))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{label}_comparison.png'), dpi=300)
            plt.close()
        else:
            print(f'Term {term} not found in any file.')

def plot_panels(panel_groups, term_labels, dfs, sizes_colors, panel_dir):
    """Plota painéis agrupando termos semelhantes."""
    for panel_name, panel_terms in panel_groups.items():
        n_terms = len(panel_terms)
        fig, axes = plt.subplots(1, n_terms, figsize=(5*n_terms, 5), sharex=True)
        if n_terms == 1:
            axes = [axes]
        for i, term in enumerate(panel_terms):
            term_key = term
            if term.endswith('_budget'):
                for k, v in term_labels.items():
                    if v == term:
                        term_key = k
                        break
            ax = axes[i]
            plotted = False
            for (size, color) in sizes_colors:
                if size in dfs and term_key in dfs[size][0].columns:
                    df, time_col = dfs[size]
                    ax.plot(df[time_col], df[term_key], label=size, color=color)
                    plotted = True
            label = term
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{label} ({get_unit(term_key, term_labels)})')
            ax.set_title(label)
            base_label = label.replace('_budget', '')
            # Limitar eixo y a 0 apenas para termos de energia (não para budget)
            if base_label in UNIT_MAP and not label.endswith('_budget'):
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %HZ'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            all_times = []
            for (size, _) in sizes_colors:
                if size in dfs and term_key in dfs[size][0].columns:
                    df, time_col = dfs[size]
                    all_times.extend(df[time_col])
            if all_times:
                ax.set_xlim(min(all_times), max(all_times))
            if i == 0:
                ax.legend(title='Domain size')
        fig.suptitle(panel_name.replace('_', ' '), fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_panel_name = panel_name.replace('/', '_').replace(' ', '_')
        fig.savefig(os.path.join(panel_dir, f'{safe_panel_name}_panel.png'), dpi=300)
        plt.close(fig)

# ===================== MAIN =====================
def main():
    """Executa o processamento e geração dos gráficos."""
    dfs = read_all_dataframes(SIZES_COLORS, BASE_DIR)
    example_size = next(iter(dfs)) if dfs else None
    if example_size:
        example_df, example_time_col = dfs[example_size]
        terms, term_labels = extract_terms_and_labels(example_df, example_time_col)
    else:
        terms, term_labels = [], {}
    plot_individual_terms(terms, term_labels, dfs, SIZES_COLORS, INDIVIDUAL_DIR)
    plot_panels(PANEL_GROUPS, term_labels, dfs, SIZES_COLORS, PANEL_DIR)

if __name__ == '__main__':
    main()
