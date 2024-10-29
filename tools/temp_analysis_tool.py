# tools/temperature_analysis_tool.py

from typing import Dict
from tools.base_tool import SingleMessageTool
import pandas as pd

class TemperatureAnalysisTool(SingleMessageTool):
    """Tool to analyze temperature data and provide insights."""

    def __init__(self, city_data: pd.DataFrame, country_data: pd.DataFrame, global_data: pd.DataFrame):
        self.city_data = city_data
        self.country_data = country_data
        self.global_data = global_data
        
        # Ensure 'Year' is extracted from 'dt' in city and country data
        for df in [self.city_data, self.country_data]:
            df['Year'] = pd.to_datetime(df['dt']).dt.year

    def get_name(self) -> str:
        return "temperature_analysis"

    def get_description(self) -> str:
        return "Analyze temperature data to provide average temperatures, trends, and comparisons."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "city": {
                "description": "The name of the city to analyze.",
                "type": "string",
                "required": False,
            },
            "country": {
                "description": "The name of the country to analyze.",
                "type": "string",
                "required": False,
            },
            "year": {
                "description": "The year to analyze.",
                "type": "integer",
                "required": False,
            },
            "analysis_type": {
                "description": "Type of analysis: 'average', 'trend', or 'comparison'.",
                "type": "string",
                "required": True,
                "enum": ["average", "trend", "comparison"]
            },
        }

    def run_impl(self, city: str = None, country: str = None, year: int = None, analysis_type: str = "average"):
        if analysis_type == "average":
            return self.calculate_average_temperature(city, country, year)
        elif analysis_type == "trend":
            return self.analyze_temperature_trend(city, country)
        elif analysis_type == "comparison":
            return self.compare_temperatures(city, country)
        else:
            return {"error": "Invalid analysis type"}

    def calculate_average_temperature(self, city: str = None, country: str = None, year: int = None):
        if city:
            city_temp_data = self.city_data[self.city_data['City'] == city]
            if year:
                city_temp_data = city_temp_data[city_temp_data['Year'] == year]
            avg_temp = city_temp_data['AverageTemperature'].mean()
            return {"average_temperature": avg_temp}
        elif country:
            country_temp_data = self.country_data[self.country_data['Country'] == country]
            if year:
                country_temp_data = country_temp_data[country_temp_data['Year'] == year]
            avg_temp = country_temp_data['AverageTemperature'].mean()
            return {"average_temperature": avg_temp}
        else:
            return {"error": "City or country must be specified"}

    def analyze_temperature_trend(self, city: str = None, country: str = None):
        if city:
            city_temp_data = self.city_data[self.city_data['City'] == city]
            trend_data = city_temp_data.groupby('Year')['AverageTemperature'].mean().reset_index()
            return trend_data.to_dict(orient='records')
        elif country:
            country_temp_data = self.country_data[self.country_data['Country'] == country]
            trend_data = country_temp_data.groupby('Year')['AverageTemperature'].mean().reset_index()
            return trend_data.to_dict(orient='records')
        else:
            return {"error": "City or country must be specified for trend analysis"}

    def compare_temperatures(self, city: str, country: str):
        city_temp_data = self.city_data[self.city_data['City'] == city]
        country_temp_data = self.country_data[self.country_data['Country'] == country]

        city_avg = city_temp_data['AverageTemperature'].mean()
        country_avg = country_temp_data['AverageTemperature'].mean()

        return {
            "city": city,
            "country": country,
            "city_average_temperature": city_avg,
            "country_average_temperature": country_avg
        }
