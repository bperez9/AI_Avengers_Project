from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import unicodedata

class TemperatureAnalysisTool(SingleMessageTool):
    """Tool to analyze temperature data and provide insights."""

    def __init__(self, city_data: pd.DataFrame):
        self.city_data = city_data.copy()

        # Ensure 'Year' is extracted from 'dt' in city data
        self.city_data['Year'] = pd.to_datetime(self.city_data['dt']).dt.year
        
        # Standardize city and country names for consistency
        self.city_data['City'] = self.city_data['City'].str.lower().str.normalize('NFKD')
        self.city_data['Country'] = self.city_data['Country'].str.lower().str.normalize('NFKD')

    def get_name(self) -> str:
        return "temperature_analysis"

    def get_description(self) -> str:
        return "Analyze temperature data to provide average temperatures, trends, comparisons, and other statistics."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "city": {
                "description": "The name of the city to analyze.",
                "type": "string",
                "required": False,
            },
            "year": {
                "description": "The year to analyze.",
                "type": "integer",
                "required": False,
            },
            "analysis_type": {
                "description": "Type of analysis: 'average', 'trend', 'comparison', 'max_temp', 'min_temp', 'missing_data'.",
                "type": "string",
                "required": True,
                "enum": ["average", "trend", "comparison", "max_temp", "min_temp", "missing_data"]
            },
        }

    def preprocess(self, city: str = None):
        """Standardize city name input for comparison."""
        if city:
            city = unicodedata.normalize('NFKD', city.lower().strip())
        return city

    def run_impl(self, city: str = None, year: int = None, analysis_type: str = "average"):
        city = self.preprocess(city)  # Preprocess city name
        
        if analysis_type == "average":
            return self.calculate_average_temperature(city, year)
        elif analysis_type == "trend":
            return self.analyze_temperature_trend(city)
        elif analysis_type == "comparison":
            return self.compare_temperatures(city)
        elif analysis_type == "max_temp":
            return self.find_max_temperature(city, year)
        elif analysis_type == "min_temp":
            return self.find_min_temperature(city, year)
        elif analysis_type == "missing_data":
            return self.analyze_missing_data(city)
        else:
            return {"error": "Invalid analysis type"}

    def calculate_average_temperature(self, city: str = None, year: int = None):
        temp_data = self.city_data
        if city:
            temp_data = temp_data[temp_data['City'] == city]
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        # Exclude rows with NaN values in 'AverageTemperature'
        temp_data = temp_data.dropna(subset=['AverageTemperature'])
        if temp_data.empty:
            return {"error": f"No valid temperature data available for {city} in {year}."}
        avg_temp = temp_data['AverageTemperature'].mean()
        return {"average_temperature": avg_temp}

    def analyze_temperature_trend(self, city: str = None):
        temp_data = self.city_data
        if city:
            temp_data = temp_data[temp_data['City'] == city]
        trend_data = temp_data.groupby('Year')['AverageTemperature'].mean().reset_index()
        return trend_data.to_dict(orient='records')

    def compare_temperatures(self, city: str):
        temp_data = self.city_data
        city_temp = temp_data[temp_data['City'] == city]
        overall_avg = temp_data['AverageTemperature'].mean()
        city_avg = city_temp['AverageTemperature'].mean()
        return {
            "city": city,
            "city_average_temperature": city_avg,
            "global_average_temperature": overall_avg,
        }

    def find_max_temperature(self, city: str = None, year: int = None):
        temp_data = self.city_data
        if city:
            temp_data = temp_data[temp_data['City'] == city]
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        max_temp = temp_data['AverageTemperature'].max()
        return {"max_temperature": max_temp}

    def find_min_temperature(self, city: str = None, year: int = None):
        temp_data = self.city_data
        if city:
            temp_data = temp_data[temp_data['City'] == city]
        if year:
            temp_data = temp_data[temp_data['Year'] == year]
        min_temp = temp_data['AverageTemperature'].min()
        return {"min_temperature": min_temp}

    def analyze_missing_data(self, city: str = None):
        temp_data = self.city_data
        if city:
            temp_data = temp_data[temp_data['City'] == city]
        missing_count = temp_data['AverageTemperature'].isna().sum()
        return {"missing_temperature_data_count": missing_count}
