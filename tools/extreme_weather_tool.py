# tools/extreme_weather_tool.py

from typing import Dict
from tools.base_tool import SingleMessageTool
from helpers.weather_helpers import (
    get_regions_with_extreme_temperatures,
    get_regions_with_heavy_rainfall,
)
# import pandas as pd

class ExtremeWeatherTool(SingleMessageTool):
    """Tool to identify regions experiencing extreme weather events."""

    def __init__(self, weather_data):
        self.weather_data = weather_data

    def get_name(self) -> str:
        return "get_regions_with_extreme_weather"

    def get_description(self) -> str:
        return "Identify regions experiencing extreme weather events such as heatwaves or heavy rainfall."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "event_type": {
                "description": "Type of extreme weather event, e.g., 'heatwave', 'heavy_rainfall'",
                "type": "string",
                "required": True,
            },
            "threshold": {
                "description": "Threshold value for the event, e.g., temperature in degrees Celsius or rainfall in mm",
                "type": "number",
                "required": True,
            },
        }

    def run_impl(self, event_type: str = None, threshold: float = None):
        results = []
        if event_type is None:
            event_types = ['heatwave', 'heavy_rainfall']
        else:
            event_types = [event_type.lower()]
        
        for event in event_types:
            if event == 'heatwave':
                if threshold is None:
                    threshold = 35.0
                regions = get_regions_with_extreme_temperatures(self.weather_data, threshold)
                results.append({
                    "event_type": "Heatwave",
                    "threshold": threshold,
                    "num_regions": len(regions),
                    "regions": regions,
                    "unit": "°C",
                    "description": f"Regions experiencing temperatures above {threshold}°C."
                })
            elif event == 'heavy_rainfall':
                if threshold is None:
                    threshold = 50.0
                regions = get_regions_with_heavy_rainfall(self.weather_data, threshold)
                results.append({
                    "event_type": "Heavy Rainfall",
                    "threshold": threshold,
                    "num_regions": len(regions),
                    "regions": regions,
                    "unit": "mm",
                    "description": f"Regions experiencing rainfall exceeding {threshold}mm."
                })
            else:
                raise ValueError(f"Unknown event_type: {event}")

        return {"results": results}

    
    # def get_regions_with_extreme_temperatures(weather_data: pd.DataFrame, threshold_temperature: float):
    #     """
    #     Identify regions where the temperature exceeds the given threshold.
    #     """
    #     # Ensure the temperature column exists
    #     if 'temperature_2m' not in weather_data.columns:
    #         raise ValueError("Column 'temperature_2m' not found in weather data.")

    #     extreme_temp_df = weather_data[weather_data['temperature_2m'] > threshold_temperature]
    #     regions = extreme_temp_df['country'].unique().tolist()
    #     return regions

    # def get_regions_with_heavy_rainfall(weather_data: pd.DataFrame, threshold_rainfall: float):
    #     """
    #     Identify regions where the precipitation exceeds the given threshold.
    #     """
    #     # Ensure the precipitation column exists
    #     if 'precipitation' not in weather_data.columns:
    #         raise ValueError("Column 'precipitation' not found in weather data.")

    #     heavy_rain_df = weather_data[weather_data['precipitation'] > threshold_rainfall]
    #     regions = heavy_rain_df['country'].unique().tolist()
    #     return regions

