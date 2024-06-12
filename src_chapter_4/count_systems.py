import os
from glob import glob
import pandas as pd

PERIODS_DIR = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods-energetics'

def abbreviate_phase_name(phase_name):
    # Define a dictionary for mapping full names to abbreviations
    abbreviations = {
        'incipient': 'Ic',
        'intensification': 'It',
        'mature': 'M',
        'decay': 'D',
        'residual': 'R',
        'incipient 2': 'Ic2',
        'intensification 2': 'It2',
        'mature 2': 'M2',
        'decay 2': 'D2'
    }
    
    # Split the phase_name on commas to get individual phases
    phases = [phase.strip() for phase in phase_name.split(',')]
    
    # Abbreviate the phases and count repetitions
    abbreviated_phases = [abbreviations[phase] for phase in phases]
    phase_counts = {phase: abbreviated_phases.count(phase) for phase in set(abbreviated_phases)}
    
    # Construct the final abbreviated name with counts if necessary
    final_name = []
    for phase in abbreviated_phases:
        if phase_counts[phase] > 1:
            final_name.append(f"{phase}{phase_counts[phase]}")
            phase_counts[phase] -= 1
        else:
            final_name.append(phase)

    return ''.join(final_name)

def main():
    regions = ['SE-BR', 'LA-PLATA', 'ARG']

    month_season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }

    for region in regions:

        # Get list of csv files
        region_dir = f'{PERIODS_DIR}/70W-no-continental_{region}' if region else f'{PERIODS_DIR}/70W-no-continental'
        csv_files = glob(f'{region_dir}/*')

        # Initialize counter for total count of systems
        total_systems_season = {'DJF': 0, 'MAM': 0, 'JJA': 0, 'SON': 0}
        seasonal_phase_counts = {season: {} for season in ['DJF', 'MAM', 'JJA', 'SON']}
        phase_counts = {}
        species_list = {}

        total_systems = len(csv_files)

        # Check if the CSV file contains the desired region in the prefix
        for csv_file in csv_files:
            if region in csv_file:
                df = pd.read_csv(csv_file, index_col=[0])
            else:
                continue 

            phases = list(df.index)
            phase_arrangement = ', '.join(phases)
            phase_counts[phase_arrangement] = phase_counts.get(phase_arrangement, 0) + 1

            # Add species to list
            cyclone_id = csv_file.split('/')[-1].split('.')[0]
            if phase_arrangement not in list(species_list.keys()):
                species_list[phase_arrangement] = []
            species_list[phase_arrangement].append(cyclone_id)

            # Get the month of the system_start
            if len(df.columns) > 0:
                system_start = pd.to_datetime(df.iloc[0][0])
                system_month = system_start.month
            else:
                continue

            # Find the corresponding season in the month_season_map
            corresponding_season = month_season_map[system_month]

            total_systems_season[corresponding_season] += 1

            # Count the seasonal occurrences of the current type beginning on the first day of the event
            seasonal_phase_counts[corresponding_season].setdefault(phase_arrangement, 0)
            seasonal_phase_counts[corresponding_season][phase_arrangement] += 1

        outdir_species = f'../results_chapter_4/species_list/'
        os.makedirs(outdir_species, exist_ok=True)

        outdir = f'../results_chapter_4/count_systems/'
        os.makedirs(outdir, exist_ok=True)

        suffix = f'_{region}' if region else '_SAt'
        
        # Export species list to CSV
        for species in species_list.keys():
            abbreviated_species = abbreviate_phase_name(species)
            species_df = pd.DataFrame(list(species_list[species]), columns=['Cyclone ID'])
            csv_name = os.path.join(outdir_species, f'{abbreviated_species}{suffix}.csv')
            species_df.to_csv(csv_name, index=False)
            print(f'{csv_name} saved.')

        # Export total count and relative percentages to CSV
        total_df = pd.DataFrame(list(phase_counts.items()), columns=['Type of System', 'Total Count'])
        total_df['Percentage'] = total_df['Total Count'] / total_systems * 100
        csv_name = os.path.join(outdir, f'total_count_of_systems{suffix}.csv')
        total_df.to_csv(csv_name, index=False)
        print(f'{csv_name} saved.')

        # Export seasonal counts and relative percentages to separate CSV files
        for season in seasonal_phase_counts.keys():
            season_df = pd.DataFrame(list(seasonal_phase_counts[season].items()), columns=['Type of System', 'Total Count'])
            season_df['Percentage'] = season_df['Total Count'] / total_systems_season[season] * 100
            csv_name = os.path.join(outdir, f'{season}_count_of_systems{suffix}.csv')
            season_df.to_csv(csv_name, index=False)
            print(f'{csv_name} saved.')

if __name__ == '__main__':
    main()