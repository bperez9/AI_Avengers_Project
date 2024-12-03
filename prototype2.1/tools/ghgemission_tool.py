from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import numpy as np

class EmissionsTool(SingleMessageTool):
    """Tool to analyze greenhouse gas emissions data."""

    def __init__(self, emissions_data):
        self.emissions_df = emissions_data
        self.emissions_df['year'] = pd.to_numeric(self.emissions_df['year'])
        self.emissions_df['value'] = pd.to_numeric(self.emissions_df['value'])

    def get_name(self) -> str:
        return "analyze_emissions_data"

    def get_description(self) -> str:
        return "Analyze global, regional, and national greenhouse gas emissions data from 1970 to 2020."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "type": "string",
                "enum": ["total_emissions", "emissions_trend", "top_emitters", "sector_analysis", "gas_composition"],
                "description": "Type of analysis to perform",
                "required": True
            },
            "start_year": {
                "type": "integer",
                "description": "Start year for analysis"
            },
            "end_year": {
                "type": "integer",
                "description": "End year for analysis"
            },
            "country": {
                "type": "string",
                "description": "ISO code of the country for analysis"
            },
            "region": {
                "type": "string",
                "description": "Region for analysis"
            },
            "sector": {
                "type": "string",
                "description": "Sector for analysis"
            },
            "gas": {
                "type": "string",
                "description": "Greenhouse gas for analysis"
            },
            "n": {
                "type": "integer",
                "description": "Number of top emitters to return"
            }
        }

    def run_impl(self, analysis_type: str, start_year: int = None, end_year: int = None, 
                 country: str = None, region: str = None, sector: str = None, 
                 gas: str = None, n: int = None):
        if analysis_type == "total_emissions":
            return self.calculate_total_emissions(start_year, end_year, country, region, sector, gas)
        elif analysis_type == "emissions_trend":
            return self.analyze_emissions_trend(start_year, end_year, country, region, sector, gas)
        elif analysis_type == "top_emitters":
            return self.identify_top_emitters(start_year, end_year, n or 10, sector, gas)
        elif analysis_type == "sector_analysis":
            return self.analyze_sector_contributions(start_year, end_year, country, region)
        elif analysis_type == "gas_composition":
            return self.analyze_gas_composition(start_year, end_year, country, region)
        else:
            return {"error": f"Unknown analysis_type: {analysis_type}"}

    def calculate_total_emissions(self, start_year, end_year, country, region, sector, gas):
        filtered_df = self.filter_data(start_year, end_year, country, region, sector, gas)
        total_emissions = filtered_df['value'].sum()
        return {"total_emissions": f"{total_emissions:.2f}"}

    def analyze_emissions_trend(self, start_year, end_year, country, region, sector, gas):
        filtered_df = self.filter_data(start_year, end_year, country, region, sector, gas)
        yearly_emissions = filtered_df.groupby('year')['value'].sum()
        trend = yearly_emissions.to_dict()
        return {"emissions_trend": trend}

    def identify_top_emitters(self, start_year, end_year, n, sector, gas):
        filtered_df = self.filter_data(start_year, end_year, sector=sector, gas=gas)
        top_emitters = filtered_df.groupby('ISO')['value'].sum().nlargest(n)
        return {"top_emitters": top_emitters.to_dict()}

    def analyze_sector_contributions(self, start_year, end_year, country, region):
        filtered_df = self.filter_data(start_year, end_year, country, region)
        sector_emissions = filtered_df.groupby('sector_title')['value'].sum()
        total_emissions = sector_emissions.sum()
        sector_contributions = (sector_emissions / total_emissions * 100).sort_values(ascending=False)
        return {"sector_contributions": sector_contributions.to_dict()}

    def analyze_gas_composition(self, start_year, end_year, country, region):
        filtered_df = self.filter_data(start_year, end_year, country, region)
        gas_emissions = filtered_df.groupby('gas')['value'].sum()
        total_emissions = gas_emissions.sum()
        gas_composition = (gas_emissions / total_emissions * 100).sort_values(ascending=False)
        return {"gas_composition": gas_composition.to_dict()}

    def filter_data(self, start_year, end_year, country=None, region=None, sector=None, gas=None):
        filtered_df = self.emissions_df.copy()
        if start_year:
            filtered_df = filtered_df[filtered_df['year'] >= start_year]
        if end_year:
            filtered_df = filtered_df[filtered_df['year'] <= end_year]
        if country:
            filtered_df = filtered_df[filtered_df['ISO'] == country]
        if region:
            filtered_df = filtered_df[filtered_df['region_ar6_6'] == region]
        if sector:
            filtered_df = filtered_df[filtered_df['sector_title'] == sector]
        if gas:
            filtered_df = filtered_df[filtered_df['gas'] == gas]
        return filtered_df