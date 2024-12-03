from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import unicodedata

class LandTemperatureAnalysisTool(SingleMessageTool):
    """Tool to analyze land temperature data and provide insights."""

    def __init__(self, land_data: pd.DataFrame):
        self.land_data = land_data.copy()

        # Ensure 'Year' and 'Month' are extracted from 'dt'
        self.land_data['Year'] = pd.to_datetime(self.land_data['dt']).dt.year
        self.land_data['Month'] = pd.to_datetime(self.land_data['dt']).dt.month
        
        # Standardize city and country names for consistency
        self.land_data['dt'] = pd.to_datetime(self.land_data['dt'])
    
    def get_name(self) -> str:
        return "land_temperature_analysis"

    def get_description(self) -> str:
        return "Analyze land temperature data to provide average temperatures, trends, comparisons, and other statistics."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "year": {
                "description": "The year to analyze.",
                "type": "integer",
                "required": False,
            },
            "month": {
                "description": "The month to analyze.",
                "type": "integer",
                "required": False,
            },
            "analysis_type": {
                "description": "Type of analysis: 'average', 'trend', 'max_temp', 'min_temp', 'compare_land_ocean'.",
                "type": "string",
                "required": True,
                "enum": ["average", "trend", "max_temp", "min_temp", "compare_land_ocean"]
            },
        }

    def preprocess(self, year: int = None, month: int = None):
        """Standardize year and month input for comparison."""
        return year, month

    def run_impl(self, year: int = None, month: int = None, analysis_type: str = "average"):
        year, month = self.preprocess(year, month)  # Preprocess inputs
        
        if analysis_type == "average":
            return self.calculate_average_temperature(year, month)
        elif analysis_type == "trend":
            return self.analyze_temperature_trend(year)
        elif analysis_type == "max_temp":
            return self.find_max_temperature(year, month)
        elif analysis_type == "min_temp":
            return self.find_min_temperature(year, month)
        elif analysis_type == "compare_land_ocean":
            return self.compare_land_ocean_temperature(year, month)
        else:
            return {"error": "Invalid analysis type"}

    def calculate_average_temperature(self, year: int = None, month: int = None):
        temp_data = self.land_data
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        if month:
            temp_data = temp_data[temp_data['Month'] == month]
        temp_data = temp_data.dropna(subset=['LandAverageTemperature'])
        if temp_data.empty:
            return {"error": "No valid temperature data available for the specified period."}
        avg_temp = temp_data['LandAverageTemperature'].mean()
        return {"average_land_temperature": avg_temp}

    def analyze_temperature_trend(self, year: int = None):
        temp_data = self.land_data
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        trend_data = temp_data.groupby('Month')['LandAverageTemperature'].mean().reset_index()
        return trend_data.to_dict(orient='records')

    def find_max_temperature(self, year: int = None, month: int = None):
        temp_data = self.land_data
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        if month:
            temp_data = temp_data[temp_data['Month'] == month]
        max_temp = temp_data['LandMaxTemperature'].max()
        return {"max_land_temperature": max_temp}

    def find_min_temperature(self, year: int = None, month: int = None):
        temp_data = self.land_data
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        if month:
            temp_data = temp_data[temp_data['Month'] == month]
        min_temp = temp_data['LandMinTemperature'].min()
        return {"min_land_temperature": min_temp}

    def compare_land_ocean_temperature(self, year: int = None, month: int = None):
        temp_data = self.land_data
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        if month:
            temp_data = temp_data[temp_data['Month'] == month]
        # Ensure both land and ocean temperatures exist
        if 'LandAndOceanAverageTemperature' not in temp_data.columns:
            return {"error": "No ocean temperature data available for comparison."}
        temp_data = temp_data.dropna(subset=['LandAndOceanAverageTemperature', 'LandAverageTemperature'])
        land_temp_avg = temp_data['LandAverageTemperature'].mean()
        land_ocean_temp_avg = temp_data['LandAndOceanAverageTemperature'].mean()
        return {
            "land_average_temperature": land_temp_avg,
            "land_ocean_average_temperature": land_ocean_temp_avg
        }
