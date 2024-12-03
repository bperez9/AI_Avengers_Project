from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

class GHGContributionTool(SingleMessageTool):
    """Tool to retrieve GHG emissions targets and mitigation contribution types for specific countries."""

    def __init__(self, contribution_data):
        self.contribution_data = contribution_data

    def get_name(self) -> str:
        return "get_ghg_emissions_targets"

    def get_description(self) -> str:
        return "Retrieve GHG emissions targets and mitigation contribution types for a specific country."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "country": {
                "description": "Country name for which to retrieve GHG emissions targets.",
                "type": "string",
                "required": True,
            },
        }

    def run_impl(self, country: str = None):
        if country is None:
            raise ValueError("Country must be specified.")

        targets = get_ghg_emissions_targets(self.contribution_data, country)

        return {"targets": targets}

def get_ghg_emissions_targets(contribution_data: pd.DataFrame, country: str):
    """
    Retrieve GHG emissions targets and mitigation contribution types for a specific country.
    """
    # Ensure the Country column exists
    if 'Country' not in contribution_data.columns:
        raise ValueError("Column 'Country' not found in contribution data.")

    # Filter data for the specified country
    country_data = contribution_data[contribution_data['Country'] == country]

    if country_data.empty:
        return {"error": f"No data found for country: {country}"}

    targets = country_data[['ghg_target', 'time_target_year', 'mitigation_contribution_type']].to_dict(orient='records')

    return targets