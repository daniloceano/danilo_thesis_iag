import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define paths
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
tracks_dir = f'{PATH}/tracks_SAt'
track_ids_path = f'{PATH}/csv_track_ids_by_region_season/all_track_ids.csv'
phase_data_path = f'../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods-energetics/70W-no-continental'
output_directory = f'../figures_chapter_5/correlation/'
merged_data_path = f'{output_directory}/merged_data.csv'

max_workers = os.cpu_count() - 1 

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}
    file_paths = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if filename.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_csv_file, file_path): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSV files"):
            file_path = futures[future]
            system_id = os.path.basename(file_path).split('_')[0]
            df = future.result()
            if df is not None:
                systems_energetics[system_id] = df

    return systems_energetics

def read_and_filter_tracks(file_path, relevant_track_ids):
    df = read_csv_file(file_path)
    df.columns = ['track_id', 'date', 'lon', 'lat', 'vor 42']
    if df is not None:
        return df[df['track_id'].isin(relevant_track_ids)]
    return pd.DataFrame()

def read_tracks(tracks_dir, relevant_track_ids):
    """
    Reads all track CSV files in the specified directory and filters relevant tracks before concatenation.
    """
    track_data = []
    file_paths = [os.path.join(tracks_dir, filename) for filename in os.listdir(tracks_dir) if filename.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_and_filter_tracks, file_path, relevant_track_ids): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading and filtering track files"):
            filtered_df = future.result()
            if not filtered_df.empty:
                track_data.append(filtered_df)

    return pd.concat(track_data, ignore_index=True) if track_data else pd.DataFrame()

def read_phase_file(file_path):
    try:
        system_id = os.path.basename(file_path).split('_')[1].split('.')[0]
        df = pd.read_csv(file_path)
        return system_id, df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def read_phase_data(phase_data_path):
    """
    Reads all phase data files in the specified directory and collects DataFrame for each system.
    """
    phase_data = {}
    file_paths = [os.path.join(phase_data_path, filename) for filename in os.listdir(phase_data_path) if filename.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_phase_file, file_path): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading phase data files"):
            system_id, df = future.result()
            if system_id and df is not None:
                phase_data[system_id] = df

    return phase_data

def assign_dates_to_single_system(system_id, df, phase_data):
    if system_id in phase_data:
        phase_dates = phase_data[system_id]
        df['date'] = None
        for _, row in phase_dates.iterrows():
            phase = row['phase']
            start_date = pd.to_datetime(row['start_date'])
            end_date = pd.to_datetime(row['end_date'])
            df.loc[df['Unnamed: 0'] == phase, 'date'] = pd.date_range(start=start_date, end=end_date, periods=len(df[df['Unnamed: 0'] == phase]))
        df['date'] = pd.to_datetime(df['date'])
    return system_id, df

def assign_dates_to_phases(systems_energetics, phase_data):
    updated_systems_energetics = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(assign_dates_to_single_system, system_id, df, phase_data): system_id for system_id, df in systems_energetics.items()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Assigning dates to phases"):
            system_id, updated_df = future.result()
            updated_systems_energetics[system_id] = updated_df
    
    return updated_systems_energetics

def merge_single_system(system_id, df, track_data):
    df['track_id'] = int(system_id)
    track_subset = track_data[track_data['track_id'] == int(system_id)]
    
    if not track_subset.empty:
        merged_df = pd.merge(track_subset, df, on=['track_id', 'date'], how='inner')
        return merged_df
    return pd.DataFrame()

def merge_data(systems_energetics, track_data):
    merged_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(merge_single_system, system_id, df, track_data): system_id for system_id, df in systems_energetics.items()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging data"):
            result = future.result()
            if not result.empty:
                merged_data.append(result)
    
    return pd.concat(merged_data, ignore_index=True) if merged_data else pd.DataFrame()


def save_merged_data(merged_data, path):
    merged_data.to_csv(path, index=False)

def main():
    # Read relevant track IDs
    relevant_track_ids = pd.read_csv(track_ids_path)['track_id'].tolist()

    # Read and process data
    systems_energetics = read_life_cycles(base_path)
    track_data = read_tracks(tracks_dir, relevant_track_ids)
    phase_data = read_phase_data(phase_data_path)

    # Assign dates to phases
    systems_energetics = assign_dates_to_phases(systems_energetics, phase_data)

    # Merge data
    merged_data = merge_data(systems_energetics, track_data)

    # Save merged data
    save_merged_data(merged_data, merged_data_path)

    print(f"Merged data saved to {merged_data_path}")

if __name__ == "__main__":
    main()
