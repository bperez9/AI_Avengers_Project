# tools/land_cover_tool.py

from typing import Dict
import pandas as pd
from Base_Tool.base_tool import SingleMessageTool

class LandCoverTool(SingleMessageTool):
    """Tool to analyze land cover data for various metrics."""

    def __init__(self, land_cover_data: pd.DataFrame):
        self.land_cover_df = land_cover_data

    def get_name(self) -> str:
        return "analyze_land_cover_data"

    def get_description(self) -> str:
        return "Analyze land cover data for urban surfaces, shrub-covered areas, and other metrics."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "type": "string",
                "enum": ["total_artificial_surfaces", "total_shrub_covered_areas"],
                "description": "Type of analysis to perform",
                "required": True
            },
            "country": {
                "type": "string",
                "description": "The name of the country, e.g., Brazil",
                "required": True
            }
        }

    def run_impl(self, analysis_type: str, country: str = None):
        # Validate the analysis type
        valid_analysis_types = {"total_artificial_surfaces", "total_shrub_covered_areas"}
        if analysis_type not in valid_analysis_types:
            raise ValueError(f"Invalid analysis type: {analysis_type}. Must be one of {valid_analysis_types}")

        # Perform the analysis based on the analysis type
        if analysis_type == "total_artificial_surfaces":
            return self.get_total_artificial_surfaces(country)
        elif analysis_type == "total_shrub_covered_areas":
            return self.get_total_shrub_covered_areas(country)

    def get_total_artificial_surfaces(self, country: str):
        country_data = self.land_cover_df[self.land_cover_df['Country'] == country]
        if country_data.empty:
            return {"error": f"No data available for {country}."}

        total_artificial_surfaces = country_data.loc[:, 'F1992':'F2022'].sum(axis=1).values[0]
        return {
            "country": country,
            "total_artificial_surfaces": f"{total_artificial_surfaces:.2f} units"
        }

    def get_total_shrub_covered_areas(self, country: str):
        country_data = self.land_cover_df[self.land_cover_df['Country'] == country]
        if country_data.empty:
            return {"error": f"No data available for {country}."}

        total_shrub_covered = country_data.loc[:, 'F1992':'F2022'].sum(axis=1).values[0]
        return {
            "country": country,
            "total_shrub_covered": f"{total_shrub_covered:.2f} units"
        }
