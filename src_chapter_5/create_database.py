import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define paths
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
PATH = '../../energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
tracks_dir = f'{PATH}/tracks_SAt'
track_ids_path = f'{PATH}/csv_track_ids_by_region_season/all_track_ids.csv'
output_directory = f'../results_chapter_5/correlation/'
merged_data_path = f'{output_directory}/merged_data.csv'

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

    with ProcessPoolExecutor(max_workers=56) as executor:
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
    df.columns = ['track_id', 'date', 'lon', 'lat', ' vor']
    if df is not None:
        return df[df['track_id'].isin(relevant_track_ids)]
    return pd.DataFrame()

def read_tracks(tracks_dir, relevant_track_ids):
    """
    Reads all track CSV files in the specified directory and filters relevant tracks before concatenation.
    """
    track_data = []
    file_paths = [os.path.join(tracks_dir, filename) for filename in os.listdir(tracks_dir) if filename.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=56) as executor:
        futures = {executor.submit(read_and_filter_tracks, file_path, relevant_track_ids): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading and filtering track files"):
            filtered_df = future.result()
            if not filtered_df.empty:
                track_data.append(filtered_df)

    return pd.concat(track_data, ignore_index=True) if track_data else pd.DataFrame()

def merge_data(systems_energetics, track_data):
    merged_data = []
    for system_id, df in systems_energetics.items():
        df['track_id'] = int(system_id)
        track_subset = track_data[track_data['track_id'] == int(system_id)]
        if not track_subset.empty:
            merged_df = pd.merge(track_subset, df, left_on='date', right_on='Unnamed: 0', how='inner')
            merged_data.append(merged_df)
    return pd.concat(merged_data, ignore_index=True)

def save_merged_data(merged_data, path):
    merged_data.to_csv(path, index=False)

def main():
    # Read relevant track IDs
    relevant_track_ids = pd.read_csv(track_ids_path)['track_id'].tolist()

    # Read and process data
    systems_energetics = read_life_cycles(base_path)
    track_data = read_tracks(tracks_dir, relevant_track_ids)

    # Merge data
    merged_data = merge_data(systems_energetics, track_data)

    # Save merged data
    save_merged_data(merged_data, merged_data_path)

    print(f"Merged data saved to {merged_data_path}")

if __name__ == "__main__":
    main()
