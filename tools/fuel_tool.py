# tools/fueltool.py

from typing import Dict
from tools.base_tool import SingleMessageTool
import pandas as pd

class FuelConsumptionTool(SingleMessageTool):
    """Tool to identify the most fuel-efficient cars based on fuel consumption in Canada."""

    def __init__(self, fuelconsumption_data):
        self.fuelconsumption_data = fuelconsumption_data

    def get_name(self) -> str:
        return "get_most_fuel_efficient_cars"

    def get_description(self) -> str:
        return "The dataset that we are using to build this function call has information about fuel consumption in Canada over the years. This function call in particular gets the information about the most fuel-efficient cars in a given year. Fuel can also mean gasoline and other synonyms. So whenever you are tasked to find the fuel efficiency of a car in a particular year we can call this function. Example: 'what is the most fuel-efficient gasoline car in 2013' we then call the function with the respective function parameters which is the year and the fuel type."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "year": {
                "description": "The year for which fuel-efficient cars need to be identified.",
                "type": "integer",
                "required": True,
            },
            "fuel_type": {
                "description": "Type of fuel for the car, e.g., 'Gasoline', 'Diesel'. If None, all fuel types are considered.",
                "type": "string",
                "required": False,
            },
        }

    def run_impl(self, year: int, fuel_type: str = None):
        if year is None:
            raise ValueError("Year must be specified.")
        if fuel_type is None:
            fuel_type = "All"  # Default fuel_type if not specified 

        print(f"Function called with year: {year}, fuel_type: {fuel_type}")

        # Filter data by year
        year_data = self.fuelconsumption_data[self.fuelconsumption_data['year'] == year]  # Use 'year' here

        if fuel_type.lower() != "all":
            # Assuming there is a column for fuel type, which you may need to add to your DataFrame.
            # Replace 'fuel_type_column_name' with the actual column name in your dataset that represents the fuel type.
            year_data = year_data[year_data['fuel_type_column_name'].str.lower() == fuel_type.lower()]  

        # Check if the data for the given year and fuel type exists
        if year_data.empty:
            raise ValueError(f"No data available for the year {year} and fuel type {fuel_type}")

        # Find the car(s) with the lowest combined fuel consumption
        # min_value = int(year_data['Combined (L/100 km)'].min())
        # new_min = min_value + 5
        most_fuel_efficient = year_data[year_data['Combined (L/100 km)'] == year_data['Combined (L/100 km)'].min() ]

        if most_fuel_efficient.empty:
            raise ValueError(f"No fuel-efficient cars found for the year {year} and fuel type {fuel_type}")

        print(f"Most fuel-efficient cars found: {most_fuel_efficient[['Make', 'Model', 'Combined (L/100 km)']].to_dict()}")

        return most_fuel_efficient[['Make', 'Model', 'Combined (L/100 km)']].to_dict()
