from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

class FutureLongtermWindTool(SingleMessageTool):
    """Tool for analyzing long-term future wind power projections from future_longterm_national.csv"""

    def __init__(self, future_longterm_data):
        """Initialize with future long-term wind power data."""
        self.future_longterm_data = future_longterm_data

    def get_name(self) -> str:
        return "analyze_future_longterm_wind"

    def get_description(self) -> str:
        return ("Analyze long-term future wind power projections to provide insights about "
                "expected capacity factors, country trends, and comparative analysis.")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "description": "Type of analysis: 'trend_analysis', 'country_projection', 'comparative_analysis', or 'peak_performance'",
                "type": "string",
                "required": True,
                "enum": ["trend_analysis", "country_projection", "comparative_analysis", "peak_performance"]
            },
            "country_code": {
                "description": "Country ISO code for specific analysis (e.g., 'DE', 'FR')",
                "type": "string",
                "required": False
            },
            "comparison_country": {
                "description": "Second country ISO code for comparison",
                "type": "string",
                "required": False
            },
            "top_n": {
                "description": "Number of top countries to return",
                "type": "integer",
                "required": False
            }
        }

    def run_impl(self, analysis_type: str, country_code: str = None, comparison_country: str = None, top_n: int = 5):
        """Implement the tool logic."""
        try:
            if analysis_type == "trend_analysis":
                return self.analyze_trends(top_n)
            elif analysis_type == "country_projection":
                if not country_code:
                    raise ValueError("Country code is required for country projection analysis")
                return self.get_country_projection(country_code)
            elif analysis_type == "comparative_analysis":
                if not country_code and comparison_country:
                    country_code = comparison_country
                elif not comparison_country and country_code:
                    raise ValueError("Second country code (comparison_country) is required for comparative analysis")
                elif not country_code and not comparison_country:
                    raise ValueError("Both country codes are required for comparative analysis")
                return self.compare_countries(country_code, comparison_country)
            elif analysis_type == "peak_performance":
                return self.analyze_peak_performance(country_code, top_n)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            raise Exception(f"Error in future long-term wind analysis: {str(e)}")

    def analyze_trends(self, top_n: int = 5):
        """Analyze overall trends in future wind power capacity."""
        try:
            self.future_longterm_data['datetime'] = pd.to_datetime(self.future_longterm_data['time'])
            
            # Get all country columns (excluding time/datetime)
            country_cols = [col for col in self.future_longterm_data.columns if col not in ['time', 'datetime']]
            
            # Calculate average capacity factors
            averages = self.future_longterm_data[country_cols].mean().sort_values(ascending=False)
            
            # Calculate overall trend (using yearly averages)
            yearly_avg = self.future_longterm_data.groupby(self.future_longterm_data['datetime'].dt.year)[country_cols].mean()
            overall_trend = yearly_avg.mean().mean()
            
            return {
                "analysis": "trend_analysis",
                "top_countries": [
                    {
                        "country_code": country,
                        "average_capacity_factor": float(value)
                    }
                    for country, value in averages.head(top_n).items()
                ],
                "overall_trend": float(overall_trend),
                "total_countries": len(country_cols)
            }
        except Exception as e:
            raise Exception(f"Error analyzing trends: {str(e)}")

    def get_country_projection(self, country_code: str):
        """Get detailed projection analysis for a specific country."""
        try:
            if country_code not in self.future_longterm_data.columns:
                raise ValueError(f"No data available for country code: {country_code}")
            
            self.future_longterm_data['datetime'] = pd.to_datetime(self.future_longterm_data['time'])
            
            # Calculate various metrics
            avg_capacity = float(self.future_longterm_data[country_code].mean())
            yearly_avg = self.future_longterm_data.groupby(self.future_longterm_data['datetime'].dt.year)[country_code].mean()
            
            return {
                "analysis": "country_projection",
                "country_code": country_code,
                "average_capacity_factor": avg_capacity,
                "yearly_projections": {
                    str(year): float(value)
                    for year, value in yearly_avg.items()
                },
                "trend_direction": "increasing" if yearly_avg.iloc[-1] > yearly_avg.iloc[0] else "decreasing"
            }
        except Exception as e:
            raise Exception(f"Error getting country projection: {str(e)}")

    def compare_countries(self, country1: str, country2: str):
        """Compare projections between two countries."""
        try:
            if country1 not in self.future_longterm_data.columns:
                raise ValueError(f"No data available for country code: {country1}")
            if country2 not in self.future_longterm_data.columns:
                raise ValueError(f"No data available for country code: {country2}")
            
            self.future_longterm_data['datetime'] = pd.to_datetime(self.future_longterm_data['time'])
            
            # Calculate averages and differences
            avg1 = float(self.future_longterm_data[country1].mean())
            avg2 = float(self.future_longterm_data[country2].mean())
            
            yearly_avg1 = self.future_longterm_data.groupby(self.future_longterm_data['datetime'].dt.year)[country1].mean()
            yearly_avg2 = self.future_longterm_data.groupby(self.future_longterm_data['datetime'].dt.year)[country2].mean()
            
            return {
                "analysis": "comparative_analysis",
                "country1": {
                    "code": country1,
                    "average_capacity_factor": avg1
                },
                "country2": {
                    "code": country2,
                    "average_capacity_factor": avg2
                },
                "difference": float(avg1 - avg2),
                "yearly_comparison": {
                    str(year): {
                        country1: float(val1),
                        country2: float(val2)
                    }
                    for (year, val1), (_, val2) in zip(yearly_avg1.items(), yearly_avg2.items())
                }
            }
        except Exception as e:
            raise Exception(f"Error comparing countries: {str(e)}")

    def analyze_peak_performance(self, country_code: str = None, top_n: int = 5):
        """Analyze peak performance periods and patterns."""
        try:
            self.future_longterm_data['datetime'] = pd.to_datetime(self.future_longterm_data['time'])
            
            if country_code:
                if country_code not in self.future_longterm_data.columns:
                    raise ValueError(f"No data available for country code: {country_code}")
                
                # Analyze specific country
                data = self.future_longterm_data[country_code]
                peak_value = float(data.max())
                peak_time = self.future_longterm_data.loc[data.idxmax(), 'datetime']
                
                return {
                    "analysis": "peak_performance",
                    "country_code": country_code,
                    "peak_capacity_factor": peak_value,
                    "peak_timestamp": str(peak_time),
                    "average_capacity_factor": float(data.mean())
                }
            else:
                # Analyze all countries
                country_cols = [col for col in self.future_longterm_data.columns if col not in ['time', 'datetime']]
                peak_averages = self.future_longterm_data[country_cols].max().sort_values(ascending=False)
                
                return {
                    "analysis": "peak_performance",
                    "top_peak_performers": [
                        {
                            "country_code": country,
                            "peak_capacity_factor": float(value)
                        }
                        for country, value in peak_averages.head(top_n).items()
                    ]
                }
        except Exception as e:
            raise Exception(f"Error analyzing peak performance: {str(e)}") 