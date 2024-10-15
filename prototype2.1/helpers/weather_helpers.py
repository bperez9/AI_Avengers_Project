# helpers/weather_helpers.py

import pandas as pd

def get_regions_with_extreme_temperatures(weather_data: pd.DataFrame, threshold_temperature: float):
    """
    Identify regions where the temperature exceeds the given threshold.
    """
    # Ensure the temperature column exists
    if 'temperature_2m' not in weather_data.columns:
        raise ValueError("Column 'temperature_2m' not found in weather data.")

    extreme_temp_df = weather_data[weather_data['temperature_2m'] > threshold_temperature]
    regions = extreme_temp_df['country'].unique().tolist()
    return regions

def get_regions_with_heavy_rainfall(weather_data: pd.DataFrame, threshold_rainfall: float):
    """
    Identify regions where the precipitation exceeds the given threshold.
    """
    # Ensure the precipitation column exists
    if 'precipitation' not in weather_data.columns:
        raise ValueError("Column 'precipitation' not found in weather data.")

    heavy_rain_df = weather_data[weather_data['precipitation'] > threshold_rainfall]
    regions = heavy_rain_df['country'].unique().tolist()
    return regions
