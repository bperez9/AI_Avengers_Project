from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

class FuelConsumptionTool(SingleMessageTool):
    """Tool to identify the most fuel-efficient cars based on fuel consumption in Canada."""

    def __init__(self, fuelconsumption_data: pd.DataFrame):
        """
        Initializes the tool with fuel consumption data.
        
        :param fuelconsumption_data: DataFrame containing the fuel consumption data for various cars.
        """
        self.fuelconsumption_data = fuelconsumption_data

    def get_name(self) -> str:
        return "get_most_fuel_efficient_cars"

    def get_description(self) -> str:
        return (
            "The dataset that we are using to build this function call has information about fuel consumption "
            "in Canada over the years. This function call in particular gets the information about the most "
            "fuel-efficient cars in a given year. Fuel can also mean gasoline and other synonyms. So whenever "
            "you are tasked to find the fuel efficiency of a car in a particular year, we can call this function. "
            "Example: 'what is the most fuel-efficient gasoline car in 2013', we then call the function with the "
            "respective function parameters which is the year and the fuel type."
        )

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
            fuel_type = "all"  # Default to include all fuel types if not specified

        # Filter data by year
        year_data = self.fuelconsumption_data[self.fuelconsumption_data['Model year'] == year]

        if fuel_type.lower() != "all":
            # Ensure the dataset contains a column for fuel type
            if 'Fuel type' not in year_data.columns:
                raise ValueError("Fuel type column not found in the dataset.")
            year_data = year_data[year_data['Fuel type'].str.lower() == fuel_type.lower()]

        # Check if data for the given year and fuel type exists
        if year_data.empty:
            return f"No data available for the year {year} and fuel type {fuel_type}"

        # Find the car(s) with the lowest combined fuel consumption
        min_combined_consumption = year_data['Combined (L/100 km)'].min()
        most_fuel_efficient = year_data[year_data['Combined (L/100 km)'] == min_combined_consumption]

        # Return the most fuel-efficient car(s) along with fuel consumption
        car_info = most_fuel_efficient[['Make', 'Model', 'Combined (L/100 km)', 'Fuel type']].to_dict(orient="records")
        
        if not car_info:
            return f"No cars found for the year {year} with the lowest fuel consumption."

        result = f"The most fuel-efficient car(s) for the year {year} are:"
        for car in car_info:
            result += f"\n{car['Make']} {car['Model']} with fuel consumption of {car['Combined (L/100 km)']} L/100 km using {car['Fuel type']}."
        
        return result

    # Additional Functions

    # Function to calculate the average CO2 emissions by car make
    def average_co2_emissions_by_make(self):
        avg_co2_emissions = self.fuelconsumption_data.groupby('Make')['CO2 emissions (g/km)'].mean()
        return avg_co2_emissions.to_dict()

    # Function to return the top 5 cars with the highest CO2 emissions
    def cars_with_highest_co2_emissions(self):   
        top_co2_emissions = self.fuelconsumption_data[['Make', 'Model', 'CO2 emissions (g/km)']].sort_values(by='CO2 emissions (g/km)', ascending=False).head(5)
        return top_co2_emissions.to_dict(orient='records')

    # Function to calculate the average engine size by vehicle class
    def average_engine_size_by_vehicle_class(self):
        avg_engine_size = self.fuelconsumption_data.groupby('Vehicle class')['Engine size (L)'].mean()
        return avg_engine_size.to_dict()

    # Function to return the top 5 most fuel-efficient cars
    def top_5_most_fuel_efficient_cars(self):
        top_fuel_efficient_cars = self.fuelconsumption_data[['Make', 'Model', 'Combined (L/100 km)']].sort_values(by='Combined (L/100 km)').head(5)
        return top_fuel_efficient_cars.to_dict(orient='records')
    
    def average_combined_fuel_efficiency_by_vehicle_class(self):
        """
        Calculates the average combined fuel efficiency by vehicle class.
        :return: Dictionary mapping vehicle classes to their average combined fuel efficiency.
        """
        if 'Vehicle class' not in self.fuelconsumption_data.columns or 'Combined (L/100 km)' not in self.fuelconsumption_data.columns:
            raise ValueError("Required columns 'Vehicle class' and 'Combined (L/100 km)' not found in the dataset.")

        avg_fuel_efficiency = self.fuelconsumption_data.groupby('Vehicle class')['Combined (L/100 km)'].mean()
        return avg_fuel_efficiency.to_dict()


