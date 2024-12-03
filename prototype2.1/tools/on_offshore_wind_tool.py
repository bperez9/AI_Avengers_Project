from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

class OnOffshoreWindTool(SingleMessageTool):
    """Tool for analyzing onshore and offshore wind power data from current_on_offshore.csv"""

    def __init__(self, onoffshore_data):
        """Initialize with onshore/offshore wind power data."""
        self.onoffshore_data = onoffshore_data

    def get_name(self) -> str:
        return "analyze_onoffshore_wind_power"

    def get_description(self) -> str:
        return ("Analyze onshore and offshore wind power data to provide insights about "
                "capacity factors, distribution, and country comparisons.")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "description": "Type of analysis: 'distribution', 'efficiency_comparison', 'top_producers', or 'country_detail'",
                "type": "string",
                "required": True,
                "enum": ["distribution", "efficiency_comparison", "top_producers", "country_detail"]
            },
            "country_code": {
                "description": "Country ISO code for specific analysis (e.g., 'DE', 'FR')",
                "type": "string",
                "required": False
            },
            "wind_type": {
                "description": "Type of wind power: 'ON' or 'OFF'",
                "type": "string",
                "required": False,
                "enum": ["ON", "OFF"]
            },
            "top_n": {
                "description": "Number of top countries to return",
                "type": "integer",
                "required": False
            }
        }

    def run_impl(self, analysis_type: str, country_code: str = None, wind_type: str = None, top_n: int = 5):
        """Implement the tool logic."""
        try:
            if analysis_type == "distribution":
                return self.analyze_distribution()
            elif analysis_type == "efficiency_comparison":
                return self.compare_efficiency(wind_type)
            elif analysis_type == "top_producers":
                return self.get_top_producers(wind_type, top_n)
            elif analysis_type == "country_detail":
                if not country_code:
                    raise ValueError("Country code is required for country detail analysis")
                return self.get_country_detail(country_code)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            raise Exception(f"Error in onshore/offshore wind analysis: {str(e)}")

    def analyze_distribution(self):
        """Analyze the distribution of onshore vs offshore wind power across countries."""
        try:
            # Convert time to datetime
            self.onoffshore_data['datetime'] = pd.to_datetime(self.onoffshore_data['time'])
            
            # Separate onshore and offshore columns
            onshore_cols = [col for col in self.onoffshore_data.columns if '_ON' in col]
            offshore_cols = [col for col in self.onoffshore_data.columns if '_OFF' in col]
            
            # Calculate average capacity factors
            onshore_avg = self.onoffshore_data[onshore_cols].mean()
            offshore_avg = self.onoffshore_data[offshore_cols].mean()
            
            return {
                "analysis": "distribution",
                "onshore_countries": len(onshore_cols),
                "offshore_countries": len(offshore_cols),
                "onshore_averages": {
                    col.split('_')[0]: float(value)
                    for col, value in onshore_avg.items()
                },
                "offshore_averages": {
                    col.split('_')[0]: float(value)
                    for col, value in offshore_avg.items()
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing distribution: {str(e)}")

    def compare_efficiency(self, wind_type: str = None):
        """Compare efficiency between onshore and offshore installations."""
        try:
            self.onoffshore_data['datetime'] = pd.to_datetime(self.onoffshore_data['time'])
            
            if wind_type:
                # Analyze specific wind type
                cols = [col for col in self.onoffshore_data.columns if f'_{wind_type}' in col]
                avg_efficiency = self.onoffshore_data[cols].mean()
                
                return {
                    "analysis": "efficiency_comparison",
                    "wind_type": wind_type,
                    "average_capacity_factor": float(avg_efficiency.mean()),
                    "country_factors": {
                        col.split('_')[0]: float(value)
                        for col, value in avg_efficiency.items()
                    }
                }
            else:
                # Compare both types
                onshore_cols = [col for col in self.onoffshore_data.columns if '_ON' in col]
                offshore_cols = [col for col in self.onoffshore_data.columns if '_OFF' in col]
                
                onshore_avg = self.onoffshore_data[onshore_cols].mean().mean()
                offshore_avg = self.onoffshore_data[offshore_cols].mean().mean()
                
                return {
                    "analysis": "efficiency_comparison",
                    "onshore_average": float(onshore_avg),
                    "offshore_average": float(offshore_avg),
                    "efficiency_difference": float(offshore_avg - onshore_avg)
                }
        except Exception as e:
            raise Exception(f"Error comparing efficiency: {str(e)}")

    def get_top_producers(self, wind_type: str = None, top_n: int = 5):
        """Get top producing countries for specified wind type or both."""
        try:
            self.onoffshore_data['datetime'] = pd.to_datetime(self.onoffshore_data['time'])
            
            if wind_type:
                cols = [col for col in self.onoffshore_data.columns if f'_{wind_type}' in col]
                averages = self.onoffshore_data[cols].mean().sort_values(ascending=False)
                
                return {
                    "analysis": "top_producers",
                    "wind_type": wind_type,
                    "top_countries": [
                        {
                            "country_code": col.split('_')[0],
                            "capacity_factor": float(value)
                        }
                        for col, value in averages.head(top_n).items()
                    ]
                }
            else:
                # Analyze both types
                onshore_cols = [col for col in self.onoffshore_data.columns if '_ON' in col]
                offshore_cols = [col for col in self.onoffshore_data.columns if '_OFF' in col]
                
                onshore_avg = self.onoffshore_data[onshore_cols].mean().sort_values(ascending=False)
                offshore_avg = self.onoffshore_data[offshore_cols].mean().sort_values(ascending=False)
                
                return {
                    "analysis": "top_producers",
                    "top_onshore": [
                        {
                            "country_code": col.split('_')[0],
                            "capacity_factor": float(value)
                        }
                        for col, value in onshore_avg.head(top_n).items()
                    ],
                    "top_offshore": [
                        {
                            "country_code": col.split('_')[0],
                            "capacity_factor": float(value)
                        }
                        for col, value in offshore_avg.head(top_n).items()
                    ]
                }
        except Exception as e:
            raise Exception(f"Error getting top producers: {str(e)}")

    def get_country_detail(self, country_code: str):
        """Get detailed analysis for a specific country."""
        try:
            onshore_col = f"{country_code}_ON"
            offshore_col = f"{country_code}_OFF"
            
            has_onshore = onshore_col in self.onoffshore_data.columns
            has_offshore = offshore_col in self.onoffshore_data.columns
            
            if not (has_onshore or has_offshore):
                raise ValueError(f"No data available for country code: {country_code}")
            
            result = {
                "analysis": "country_detail",
                "country_code": country_code,
                "has_onshore": has_onshore,
                "has_offshore": has_offshore
            }
            
            if has_onshore:
                onshore_avg = float(self.onoffshore_data[onshore_col].mean())
                result["onshore_capacity_factor"] = onshore_avg
            
            if has_offshore:
                offshore_avg = float(self.onoffshore_data[offshore_col].mean())
                result["offshore_capacity_factor"] = offshore_avg
            
            if has_onshore and has_offshore:
                result["capacity_factor_difference"] = float(offshore_avg - onshore_avg)
            
            return result
        except Exception as e:
            raise Exception(f"Error getting country detail: {str(e)}") 