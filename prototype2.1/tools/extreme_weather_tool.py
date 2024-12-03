# tools/extreme_weather_tool.py

from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

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
                "description": "Type of extreme weather event: 'heatwave', 'heavy_rainfall', 'high_humidity', 'strong_winds', 'uv_warning', 'soil_analysis'",
                "type": "string",
                "required": True,
            },
            "threshold": {
                "description": "Threshold value for the event (if applicable)",
                "type": "number",
                "required": False,
            },
        }

    def get_regions_with_extreme_temperatures(self, weather_data: pd.DataFrame, threshold_temperature: float):
        """
        Identify regions where the temperature exceeds the given threshold.
        """
        # Ensure the temperature column exists
        if 'temperature_2m' not in weather_data.columns:
            raise ValueError("Column 'temperature_2m' not found in weather data.")

        extreme_temp_df = weather_data[weather_data['temperature_2m'] > threshold_temperature]
        regions = extreme_temp_df['country'].unique().tolist()
        return regions

    def get_regions_with_heavy_rainfall(self, weather_data: pd.DataFrame, threshold_rainfall: float):
        """
        Identify regions where the precipitation exceeds the given threshold.
        """
        # Ensure the precipitation column exists
        if 'precipitation' not in weather_data.columns:
            raise ValueError("Column 'precipitation' not found in weather data.")

        heavy_rain_df = weather_data[weather_data['precipitation'] > threshold_rainfall]
        regions = heavy_rain_df['country'].unique().tolist()
        return regions

    def get_regions_with_high_humidity(self, weather_data: pd.DataFrame, threshold_humidity: float = 90):
        """
        Identify regions with very high humidity levels (default >90%).
        Returns regions and their maximum humidity levels.
        """
        high_humidity_df = weather_data[weather_data['relative_humidity_2m'] > threshold_humidity]
        if high_humidity_df.empty:
            return []
        
        # Group by country and get max humidity
        humidity_by_region = high_humidity_df.groupby('country')['relative_humidity_2m'].max()
        return humidity_by_region.to_dict()

    def analyze_wind_conditions(self, weather_data: pd.DataFrame, high_wind_threshold: float = 20):
        """
        Analyze wind conditions to identify regions with strong winds and gusts.
        Returns dictionary with high wind events and their details.
        """
        high_wind_df = weather_data[
            (weather_data['wind_speed_10m'] > high_wind_threshold) |
            (weather_data['wind_gusts_10m'] > high_wind_threshold * 1.5)
        ]
        
        if high_wind_df.empty:
            return {}
        
        wind_analysis = {
            country: {
                'max_wind_speed': float(group['wind_speed_10m'].max()),
                'max_wind_gusts': float(group['wind_gusts_10m'].max()),
                'predominant_direction': int(group['wind_direction_10m'].mode().iloc[0])
            }
            for country, group in high_wind_df.groupby('country')
        }
        
        return wind_analysis

    def get_uv_index_warnings(self, weather_data: pd.DataFrame, high_uv_threshold: float = 8):
        """
        Identify regions with dangerous UV index levels.
        Returns regions with high UV exposure risk.
        """
        high_uv_df = weather_data[weather_data['uv_index'] > high_uv_threshold]
        if high_uv_df.empty:
            return {}
        
        uv_analysis = {
            country: {
                'max_uv_index': float(group['uv_index'].max()),
                'max_uv_index_clear_sky': float(group['uv_index_clear_sky'].max()),
                'exposure_hours': int(len(group))
            }
            for country, group in high_uv_df.groupby('country')
        }
        
        return uv_analysis

    def analyze_soil_conditions(self, weather_data: pd.DataFrame):
        """
        Analyze soil conditions with minimal output.
        Returns a simplified analysis of soil temperature and moisture.
        """
        soil_analysis = {}
        
        for country, group in weather_data.groupby('country'):
            latest_data = group.iloc[-1]
            
            # Simple temperature and moisture indicators
            temp_status = "warm" if float(latest_data['soil_temperature_0cm']) > 20 else "cool"
            moisture = float(latest_data['soil_moisture_0_to_1cm'])
            moisture_status = "dry" if moisture < 0.2 else "wet" if moisture > 0.25 else "normal"
            
            soil_analysis[country] = f"{temp_status}, {moisture_status}"
        
        return soil_analysis

    def run_impl(self, event_type: str = None, threshold: float = None):
        results = []
        if event_type is None:
            event_types = ['heatwave', 'heavy_rainfall', 'high_humidity', 'strong_winds', 'uv_warning', 'soil_analysis']
        else:
            event_types = [event_type.lower()]
        
        for event in event_types:
            if event == 'heatwave':
                if threshold is None:
                    threshold = 35.0
                regions = self.get_regions_with_extreme_temperatures(self.weather_data, threshold)
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
                regions = self.get_regions_with_heavy_rainfall(self.weather_data, threshold)
                results.append({
                    "event_type": "Heavy Rainfall",
                    "threshold": threshold,
                    "num_regions": len(regions),
                    "regions": regions,
                    "unit": "mm",
                    "description": f"Regions experiencing rainfall exceeding {threshold}mm."
                })
            elif event == 'high_humidity':
                threshold = threshold or 90.0
                regions = self.get_regions_with_high_humidity(self.weather_data, threshold)
                results.append({
                    "event_type": "High Humidity",
                    "threshold": threshold,
                    "regions": regions,
                    "unit": "%",
                    "description": f"Regions experiencing humidity levels above {threshold}%"
                })
            elif event == 'strong_winds':
                threshold = threshold or 20.0
                wind_data = self.analyze_wind_conditions(self.weather_data, threshold)
                results.append({
                    "event_type": "Strong Winds",
                    "threshold": threshold,
                    "conditions": wind_data,
                    "unit": "m/s",
                    "description": f"Regions experiencing wind speeds above {threshold} m/s"
                })
            elif event == 'uv_warning':
                threshold = threshold or 8.0
                uv_data = self.get_uv_index_warnings(self.weather_data, threshold)
                results.append({
                    "event_type": "UV Warning",
                    "threshold": threshold,
                    "warnings": uv_data,
                    "description": f"Regions with UV index above {threshold}"
                })
            elif event == 'soil_analysis':
                soil_data = self.analyze_soil_conditions(self.weather_data)
                results.append({
                    "event_type": "Soil Analysis",
                    "conditions": soil_data,
                    "description": "Comprehensive soil temperature and moisture analysis by region"
                })
            else:
                raise ValueError(f"Unknown event_type: {event}")

        return {"results": results}
