from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd

class WindNationalTool(SingleMessageTool):
    """Tool for analyzing national wind power data from current_national_1.csv"""

    def __init__(self, wind_data):
        """Initialize with wind power data."""
        self.wind_data = wind_data
        self.country_iso_map = {
            'Albania': 'AL',
            'Austria': 'AT',
            'Bosnia and Herzegovina': 'BA',
            'Belgium': 'BE',
            'Bulgaria': 'BG',
            'Switzerland': 'CH',
            'Cyprus': 'CY',
            'Czech Republic': 'CZ',
            'Germany': 'DE',
            'Denmark': 'DK',
            'Estonia': 'EE',
            'Greece': 'EL',
            'Spain': 'ES',
            'Finland': 'FI',
            'France': 'FR',
            'Croatia': 'HR',
            'Hungary': 'HU',
            'Ireland': 'IE',
            'Italy': 'IT',
            'Lithuania': 'LT',
            'Luxembourg': 'LU',
            'Latvia': 'LV',
            'Moldova': 'MD',
            'Montenegro': 'ME',
            'Macedonia': 'MK',
            'Malta': 'MT',
            'Netherlands': 'NL',
            'Norway': 'NO',
            'Poland': 'PL',
            'Portugal': 'PT',
            'Romania': 'RO',
            'Serbia': 'RS',
            'Sweden': 'SE',
            'Slovenia': 'SI',
            'Slovakia': 'SK',
            'United Kingdom': 'GB',
        }

    def get_name(self) -> str:
        return "analyze_national_wind_power"

    def get_description(self) -> str:
        return ("Analyze national wind power data to provide insights about capacity factors, "
                "seasonal patterns, and country comparisons.")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "description": "Type of analysis: 'top_performers', 'seasonal_pattern', or 'country_comparison'",
                "type": "string",
                "required": True,
                "enum": ["top_performers", "seasonal_pattern", "country_comparison"]
            },
            "country": {
                "description": "Country name for specific analysis (required for country_comparison)",
                "type": "string",
                "required": False
            },
            "season": {
                "description": "Season number (1-4) for seasonal analysis",
                "type": "integer",
                "required": False
            },
            "top_n": {
                "description": "Number of top countries to return",
                "type": "integer",
                "required": False
            }
        }

    def run_impl(self, analysis_type: str, country: str = None, season: int = None, top_n: int = 5):
        """Implement the tool logic."""
        try:
            if analysis_type == "top_performers":
                return self.get_top_performing_countries(top_n)
            elif analysis_type == "seasonal_pattern":
                if season and season not in [1, 2, 3, 4]:
                    raise ValueError("Season must be between 1 and 4")
                return self.analyze_seasonal_patterns(season)
            elif analysis_type == "country_comparison":
                if not country:
                    raise ValueError("Country is required for country comparison")
                return self.compare_country_performance(country)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            raise Exception(f"Error in wind power analysis: {str(e)}")

    def get_top_performing_countries(self, top_n: int = 5):
        """Get the top performing countries based on average capacity factor."""
        try:
            # Convert time to datetime
            self.wind_data['datetime'] = pd.to_datetime(self.wind_data['time'])
            
            # Get numeric columns (country data)
            numeric_cols = self.wind_data.select_dtypes(include=['float64', 'int64']).columns
            
            # Calculate average capacity factor for each country
            country_averages = self.wind_data[numeric_cols].mean().sort_values(ascending=False)
            
            # Get top N countries
            top_countries = country_averages.head(top_n)
            
            return {
                "analysis": "top_performers",
                "results": [
                    {
                        "country": country,
                        "average_capacity_factor": float(value)
                    }
                    for country, value in top_countries.items()
                ]
            }
        except Exception as e:
            raise Exception(f"Error calculating top performers: {str(e)}")

    def analyze_seasonal_patterns(self, season: int = None):
        """Analyze seasonal patterns in wind power capacity."""
        try:
            # Convert time to datetime and extract season
            self.wind_data['datetime'] = pd.to_datetime(self.wind_data['time'])
            self.wind_data['season'] = self.wind_data['datetime'].dt.month % 12 // 3 + 1
            
            # Get numeric columns (country data)
            numeric_cols = self.wind_data.select_dtypes(include=['float64', 'int64']).columns
            
            if season:
                # Analyze specific season
                seasonal_data = self.wind_data[self.wind_data['season'] == season]
                seasonal_avg = seasonal_data[numeric_cols].mean()
                
                return {
                    "analysis": "seasonal_pattern",
                    "season": season,
                    "results": {
                        country: float(value)
                        for country, value in seasonal_avg.items()
                    }
                }
            else:
                # Analyze all seasons
                seasonal_avg = self.wind_data.groupby('season')[numeric_cols].mean()
                
                return {
                    "analysis": "seasonal_pattern",
                    "results": {
                        int(season): {
                            country: float(value)
                            for country, value in row.items()
                        }
                        for season, row in seasonal_avg.iterrows()
                    }
                }
        except Exception as e:
            raise Exception(f"Error analyzing seasonal patterns: {str(e)}")

    def compare_country_performance(self, country: str):
        """Compare a country's performance with others."""
        try:
            if country not in self.country_iso_map:
                raise ValueError(f"Unknown country: {country}")
            
            # Convert time to datetime
            self.wind_data['datetime'] = pd.to_datetime(self.wind_data['time'])
            
            # Get numeric columns (country data)
            numeric_cols = self.wind_data.select_dtypes(include=['float64', 'int64']).columns
            
            # Calculate average capacity factors
            country_averages = self.wind_data[numeric_cols].mean()
            overall_avg = country_averages.mean()
            country_value = country_averages[self.country_iso_map[country]]
            
            # Calculate ranking
            ranking = (country_averages > country_value).sum() + 1
            
            return {
                "analysis": "country_comparison",
                "country": country,
                "capacity_factor": float(country_value),
                "overall_average": float(overall_avg),
                "ranking": int(ranking),
                "total_countries": len(country_averages),
                "performance_vs_avg": float(country_value - overall_avg)
            }
        except Exception as e:
            raise Exception(f"Error comparing country performance: {str(e)}") 