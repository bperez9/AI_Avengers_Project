# tools/sea_level_tool.py

from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import numpy as np
from scipy import stats

class SeaLevelTool(SingleMessageTool):
    """Tool to analyze sea level and GMSL data."""

    def __init__(self, sea_level_data: pd.DataFrame, gsml_data: pd.DataFrame):
        # Initialize with sea level and GSML data
        self.sea_level_df = sea_level_data
        self.gsml_df = gsml_data
        # Convert 'Year' and 'Time' columns to datetime format
        self.sea_level_df['Year'] = pd.to_datetime(self.sea_level_df['Year'], errors='coerce')
        self.gsml_df['Time'] = pd.to_datetime(self.gsml_df['Time'], errors='coerce')

    def get_name(self) -> str:
        return "analyze_sea_level_data"

    def get_description(self) -> str:
        return "Analyze sea level and Global Mean Sea Level (GMSL) data for various metrics and projections."

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "type": "string",
                "enum": ["rise_rate", "projection", "anomalies", "average_gmsl", "max_uncertainty"],
                "description": "Type of analysis to perform",
                "required": True
            },
            "start_year": {
                "type": "integer",
                "description": "Start year for analysis"
            },
            "end_year": {
                "type": "integer",
                "description": "End year for analysis or projection target year"
            },
            "n": {
                "type": "integer",
                "description": "Number of top anomalies to return"
            }
        }

    def run_impl(self, analysis_type: str, start_year: int = None, end_year: int = None, n: int = None):
         # Define valid analysis types
        valid_analysis_types = {"rise_rate", "projection", "anomalies", "average_gmsl", "max_uncertainty"}
    
    # Raise ValueError for invalid analysis types
        if analysis_type not in valid_analysis_types:
            raise ValueError(f"Invalid analysis type: {analysis_type}. Must be one of {valid_analysis_types}")
    
    # Perform the analysis based on the analysis type
        if analysis_type == "rise_rate":
            return self.calculate_average_sea_level_rise_rate(start_year, end_year)
        elif analysis_type == "projection":
            return self.project_sea_level(end_year)
        elif analysis_type == "anomalies":
            return self.top_sea_level_anomaly_years(n or 5, start_year, end_year)
        elif analysis_type == "average_gmsl":
            return self.average_gmsl_for_year(start_year)
        elif analysis_type == "max_uncertainty":
            return self.max_gmsl_uncertainty(start_year, end_year)


    def calculate_average_sea_level_rise_rate(self, start_year: int, end_year: int):
        mask = (self.sea_level_df['Year'].dt.year >= start_year) & (self.sea_level_df['Year'].dt.year <= end_year)
        filtered_df = self.sea_level_df.loc[mask]
        if filtered_df.empty:
            return {"error": "No data available for the specified time range"}
        total_rise = filtered_df['CSIRO Adjusted Sea Level'].iloc[-1] - filtered_df['CSIRO Adjusted Sea Level'].iloc[0]
        years = end_year - start_year
        rate = total_rise / years
        return {"average_rise_rate": f"{rate:.4f} inches per year"}

    def project_sea_level(self, target_year: int):
        x = self.sea_level_df['Year'].dt.year
        y = self.sea_level_df['CSIRO Adjusted Sea Level']
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) == 0 or len(y) == 0:
            return {"error": "No valid data for calculation"}
        slope, intercept, _, _, _ = stats.linregress(x, y)
        if np.isnan(slope) or np.isnan(intercept):
            return {"error": "Unable to calculate regression"}
        projected_level = slope * target_year + intercept
        return {"projected_sea_level": f"{projected_level:.2f} inches"}

    def top_sea_level_anomaly_years(self, n: int, start_year: int, end_year: int):
        mask = (self.sea_level_df['Year'].dt.year >= start_year) & (self.sea_level_df['Year'].dt.year <= end_year)
        filtered_df = self.sea_level_df.loc[mask].copy()
        if filtered_df.empty:
            return {"error": "No data available for the specified time range"}
        mean_level = filtered_df['CSIRO Adjusted Sea Level'].mean()
        filtered_df['Anomaly'] = filtered_df['CSIRO Adjusted Sea Level'] - mean_level
        top_anomalies = filtered_df.nlargest(n, 'Anomaly')[['Year', 'Anomaly']]
        top_anomalies['Year'] = top_anomalies['Year'].dt.strftime('%Y-%m-%d')
        top_anomalies['Anomaly'] = top_anomalies['Anomaly'].round(4)
        return {"top_anomalies": top_anomalies.to_dict(orient='records')}

    def average_gmsl_for_year(self, year: int):
        yearly_data = self.gsml_df[self.gsml_df['Time'].dt.year == year]
        if yearly_data.empty:
            return {"error": f"No data available for the year {year}"}
        avg_gmsl = yearly_data['GMSL'].mean()
        return {"average_gmsl": f"{avg_gmsl:.2f} mm"}

    def max_gmsl_uncertainty(self, start_year: int, end_year: int):
        mask = (self.gsml_df['Time'].dt.year >= start_year) & (self.gsml_df['Time'].dt.year <= end_year)
        filtered_df = self.gsml_df.loc[mask]
        if filtered_df.empty:
            return {"error": "No data available for the specified time range"}
        max_uncertainty = filtered_df['GMSL uncertainty'].max()
        if pd.isna(max_uncertainty):
            return {"error": "Maximum uncertainty is NaN"}
        return {"max_uncertainty": f"{max_uncertainty:.2f} mm"}
