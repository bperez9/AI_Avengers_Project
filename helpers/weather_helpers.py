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
    Identify regions where the daily total precipitation exceeds the given threshold.
    """
    # Ensure necessary columns exist
    required_columns = ['precipitation', 'time', 'country']
    for col in required_columns:
        if col not in weather_data.columns:
            raise ValueError(f"Column '{col}' not found in weather data.")

    # Convert 'time' column to datetime
    weather_data['time'] = pd.to_datetime(weather_data['time'])

    # Add a 'date' column
    weather_data['date'] = weather_data['time'].dt.date

    # Group by 'country' and 'date', then sum the 'precipitation'
    daily_precipitation = weather_data.groupby(['country', 'date'])['precipitation'].sum().reset_index()

    # Identify days where total precipitation exceeds the threshold
    heavy_rain_df = daily_precipitation[daily_precipitation['precipitation'] > threshold_rainfall]

    # Get the unique countries
    regions = heavy_rain_df['country'].unique().tolist()
    return regions

