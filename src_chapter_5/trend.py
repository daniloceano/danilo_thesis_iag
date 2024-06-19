import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import gaussian_kde, kendalltau
from matplotlib.colors import BoundaryNorm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
tracks_dir = f'{PATH}/tracks_SAt'
track_ids_path = f'{PATH}/csv_track_ids_by_region_season/all_track_ids.csv'
output_directory = f'../figures_chapter_5/correlation/'

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

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_csv_file, file_path): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSV files"):
            file_path = futures[future]
            system_id = os.path.basename(file_path).split('_')[0]
            df = future.result()
            if df is not None:
                systems_energetics[system_id] = df

    return systems_energetics

def read_tracks(tracks_dir, relevant_track_ids):
    """
    Reads all track CSV files in the specified directory and filters relevant tracks.
    """
    track_data = pd.DataFrame()
    file_paths = [os.path.join(tracks_dir, filename) for filename in os.listdir(tracks_dir) if filename.endswith('.csv')]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_csv_file, file_path): file_path for file_path in file_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading track files"):
            df = future.result()
            if df is not None:
                track_data = pd.concat([track_data, df], ignore_index=True)

    return track_data[track_data['track_id'].isin(relevant_track_ids)]

def compute_mean_values(systems_energetics):
    mean_values = {}
    for system_id, df in systems_energetics.items():
        numeric_df = df.select_dtypes(include=[np.number])
        mean_energetics = numeric_df.mean()
        mean_values[system_id] = mean_energetics
    return pd.DataFrame(mean_values).T

def visualize_time_series(df, terms):
    for term in terms:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[term], label=term)
        plt.title(f'Time Series of {term}')
        plt.xlabel('Time')
        plt.ylabel(term)
        plt.legend()
        plt.show()

def kendall_trend_test(df, terms):
    for term in terms:
        df_term = df[term].dropna()
        tau, p_value = kendalltau(df_term.index, df_term.values)
        trend = "increasing" if tau > 0 else "decreasing" if tau < 0 else "no trend"
        print(f'{term}: {trend}, p-value: {p_value}')

def linear_trend_analysis(df, term):
    df = df.dropna(subset=[term])
    X = np.array((df.index - df.index[0]).days).reshape(-1, 1)  # Convert dates to ordinal
    y = df[term].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    return trend

def analyze_linear_trends(df, terms):
    for term in terms:
        trend = linear_trend_analysis(df, term)
        print(f'Trend for {term}: {trend[0]} per day')

def seasonal_decomposition_analysis(df, terms):
    for term in terms:
        result = seasonal_decompose(df[term].dropna(), model='additive', period=365)
        result.plot()
        plt.suptitle(f'Seasonal Decomposition of {term}')
        plt.show()

def arima_model(df, term, order=(1, 1, 1)):
    df = df.dropna(subset=[term])
    model = ARIMA(df[term], order=order)
    result = model.fit()
    return result

def arima_analysis(df, terms):
    for term in terms:
        result = arima_model(df, term)
        print(f'ARIMA model summary for {term}:\n{result.summary()}')
        result.plot_predict(start=1, end=len(df) + 10)
        plt.title(f'ARIMA Model Prediction for {term}')
        plt.show()

def main():
    # Read relevant track IDs
    relevant_track_ids = pd.read_csv(track_ids_path)['track_id'].tolist()

    # Read and process data
    systems_energetics = read_life_cycles(base_path)
    track_data = read_tracks(tracks_dir, relevant_track_ids)

    # Extract date information and add to energetics data
    all_data = []
    for system_id, df in systems_energetics.items():
        track_dates = track_data[track_data['track_id'] == int(system_id)]['date'].values
        df['date'] = pd.to_datetime(track_dates)
        df['system_id'] = system_id
        all_data.append(df)

    energetics_df = pd.concat(all_data, ignore_index=True)
    energetics_df.set_index('date', inplace=True)

    # Compute mean values for each system
    mean_values_df = compute_mean_values(systems_energetics)

    # Terms to analyze
    terms = ['Az', 'Ae', 'Kz', 'Cz', 'BAz', 'BAe', 'BKz', 'BKe', 'Gz', 'Ge', 'RGz', 'RKz', 'RGe', 'RKe']

    # Visualization
    visualize_time_series(mean_values_df, terms)

    # Statistical Tests
    kendall_trend_test(mean_values_df, terms)

    # Regression Analysis
    analyze_linear_trends(mean_values_df, terms)

    # Seasonal Decomposition
    seasonal_decomposition_analysis(mean_values_df, terms)

    # ARIMA Analysis
    arima_analysis(mean_values_df, terms)

if __name__ == "__main__":
    main()
