import pandas as pd
import os

# Define the paths
base_path = '../../Programs_and_scripts/LEC_Results_energetic-patterns/'
track_base_path = '../../Programs_and_scripts/SWSA-cyclones_energetic-analysis/raw_data/SAt/'

# Function to label phases based on periods
def label_phases(row, periods_df):
    for _, period in periods_df.iterrows():
        if period['start'] <= row['date'] <= period['end']:
            return period.values[0]
    return 'No Phase'

# Function to process each cyclone directory
def process_cyclone(cyclone_dir):
    try:
        # Extract cyclone ID from directory name
        track_id = cyclone_dir.split('_')[0]
        
        # Load the energy results
        energy_file = os.path.join(base_path, cyclone_dir, f'{track_id}_ERA5_track_results.csv')
        if not os.path.exists(energy_file):
            return
        energy_results = pd.read_csv(energy_file, header=0)
        energy_results.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        energy_results['date'] = pd.to_datetime(energy_results['date'])
        
        # Extract start year and month from the first timestamp in the energy results
        start_time = energy_results['date'].iloc[0]
        year_month = start_time.strftime('%Y%m')
        
        # Load the track data for the corresponding year and month
        track_file = os.path.join(track_base_path, f'ff_cyc_SAt_era5_{year_month}.csv')
        track_data = pd.read_csv(track_file)
        track_data.columns = ['track_id', 'date', 'lon', 'lat', 'vor 42']
        track_data['date'] = pd.to_datetime(track_data['date'])
        
        # Filter track data for the specific cyclone
        cyclone_track_data = track_data[track_data['track_id'] == int(track_id)]
        
        # Load the specific periods data for this cyclone
        periods_file = os.path.join(base_path, cyclone_dir, 'periods.csv')
        periods = pd.read_csv(periods_file, header=0)
        periods['start'] = pd.to_datetime(periods['start'])
        periods['end'] = pd.to_datetime(periods['end'])
        
        # Merge track data and energy results on the appropriate keys
        merged_data = pd.merge(cyclone_track_data, energy_results, on='date')
        
        # Apply the labeling function to the merged data
        merged_data['phase'] = merged_data.apply(label_phases, periods_df=periods, axis=1)
        
        # Save the final merged data to a new CSV file
        output_file = f'/mnt/data/merged_cyclone_data_{track_id}.csv'
        merged_data.to_csv(output_file, index=False)
        
        print(f"Processed cyclone {track_id} successfully and saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing cyclone {track_id}: {e}")

# Iterate over all cyclone directories in the base path
for cyclone_dir in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, cyclone_dir)):
        process_cyclone(cyclone_dir)
