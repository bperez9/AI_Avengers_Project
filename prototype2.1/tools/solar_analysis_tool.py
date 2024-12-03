from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import numpy as np

class SolarAnalysisTool(SingleMessageTool):
    """Tool for analyzing solar power data from both SARAH and MERRA datasets"""

    def __init__(self, sarah_data, merra_data):
        """Initialize with both SARAH and MERRA solar data."""
        self.sarah_data = sarah_data
        self.merra_data = merra_data

    def get_name(self) -> str:
        return "analyze_solar_data"

    def get_description(self) -> str:
        return ("Analyze solar power data using either SARAH (specialized for solar energy) "
                "or MERRA (broader environmental context) datasets.")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "description": ("Type of analysis: 'daylight_patterns', 'geographical_patterns', "
                              "'clear_sky_patterns', 'country_analysis', 'regional_comparison', "
                              "'seasonal_efficiency'"),
                "type": "string",
                "required": True,
                "enum": [
                    "daylight_patterns", "geographical_patterns", "clear_sky_patterns",
                    "country_analysis", "regional_comparison", "seasonal_efficiency"
                ]
            },
            "data_source": {
                "description": "Source dataset to use: 'sarah' (solar-specific) or 'merra' (environmental context)",
                "type": "string",
                "required": True,
                "enum": ["sarah", "merra"]
            },
            "country_code": {
                "description": "Country ISO code for analysis (e.g., 'DE', 'FR')",
                "type": "string",
                "required": False
            },
            "comparison_country": {
                "description": "Second country ISO code for comparison",
                "type": "string",
                "required": False
            }
        }

    def run_impl(self, analysis_type: str, data_source: str, country_code: str = None, 
                 comparison_country: str = None):
        """Implement the tool logic."""
        try:
            # Select the appropriate dataset
            data = self.sarah_data if data_source.lower() == "sarah" else self.merra_data
            
            if analysis_type == "daylight_patterns":
                return self.analyze_daylight_patterns(data, country_code)
            elif analysis_type == "geographical_patterns":
                return self.analyze_geographical_patterns(data)
            elif analysis_type == "clear_sky_patterns":
                return self.analyze_clear_sky_patterns(data, country_code)
            elif analysis_type == "country_analysis":
                if not country_code:
                    raise ValueError("Country code is required for country analysis")
                return self.analyze_country(data, country_code)
            elif analysis_type == "regional_comparison":
                if not country_code or not comparison_country:
                    raise ValueError("Both country codes are required for comparison")
                return self.compare_regions(data, country_code, comparison_country)
            elif analysis_type == "seasonal_efficiency":
                return self.analyze_seasonal_efficiency(data, country_code)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            raise Exception(f"Error in solar analysis: {str(e)}")

    def analyze_daylight_patterns(self, data: pd.DataFrame, country_code: str = None):
        """Analyze daylight hours and solar intensity patterns."""
        try:
            data['datetime'] = pd.to_datetime(data['time'])
            if country_code:
                if country_code not in data.columns:
                    raise ValueError(f"No data available for country code: {country_code}")
                solar_data = data[country_code]
            else:
                solar_data = data.drop(['time', 'datetime'], axis=1).mean(axis=1)
            
            # Calculate daylight hours (any period with non-zero solar radiation)
            daily_daylight = solar_data[solar_data > 0].groupby(data['datetime'].dt.date).count()
            
            # Calculate seasonal daylight averages
            seasonal_daylight = {
                "summer": float(solar_data[data['datetime'].dt.month.isin([6,7,8])].groupby(data['datetime'].dt.date).count().mean()),
                "winter": float(solar_data[data['datetime'].dt.month.isin([12,1,2])].groupby(data['datetime'].dt.date).count().mean())
            }
            
            # Calculate sunrise and sunset times
            daily_first_light = data[solar_data > 0].groupby(data['datetime'].dt.date)['datetime'].min()
            daily_last_light = data[solar_data > 0].groupby(data['datetime'].dt.date)['datetime'].max()
            
            # Convert times to minutes for averaging
            avg_sunrise_minutes = daily_first_light.dt.hour * 60 + daily_first_light.dt.minute
            avg_sunset_minutes = daily_last_light.dt.hour * 60 + daily_last_light.dt.minute
            
            # Calculate average times
            avg_sunrise = pd.Timestamp('today').replace(
                hour=int(avg_sunrise_minutes.mean() // 60),
                minute=int(avg_sunrise_minutes.mean() % 60)
            ).time()
            
            avg_sunset = pd.Timestamp('today').replace(
                hour=int(avg_sunset_minutes.mean() // 60),
                minute=int(avg_sunset_minutes.mean() % 60)
            ).time()
            
            return {
                "analysis": "daylight_patterns",
                "daylight_metrics": {
                    "average_daylight_hours": float(daily_daylight.mean()),
                    "max_daylight_hours": float(daily_daylight.max()),
                    "min_daylight_hours": float(daily_daylight.min())
                },
                "timing_metrics": {
                    "average_sunrise": str(avg_sunrise),
                    "average_sunset": str(avg_sunset),
                    "earliest_sunrise": str(daily_first_light.min().time()),
                    "latest_sunset": str(daily_last_light.max().time())
                },
                "seasonal_variation": seasonal_daylight
            }
        except Exception as e:
            raise Exception(f"Error analyzing daylight patterns: {str(e)}")

    def analyze_geographical_patterns(self, data: pd.DataFrame):
        """Analyze solar patterns based on geographical location."""
        try:
            data['datetime'] = pd.to_datetime(data['time'])
            
            # Define geographical groups
            northern_countries = ['NO', 'SE', 'FI', 'DK']
            central_countries = ['DE', 'FR', 'PL', 'CZ']
            southern_countries = ['ES', 'IT', 'GR', 'PT']
            
            # Calculate regional averages
            north_data = data[northern_countries].mean(axis=1)
            central_data = data[central_countries].mean(axis=1)
            south_data = data[southern_countries].mean(axis=1)
            
            return {
                "analysis": "geographical_patterns",
                "regional_averages": {
                    "northern_europe": float(north_data.mean()),
                    "central_europe": float(central_data.mean()),
                    "southern_europe": float(south_data.mean())
                },
                "seasonal_patterns": {
                    "summer": {
                        "northern": float(north_data[data['datetime'].dt.month.isin([6,7,8])].mean()),
                        "central": float(central_data[data['datetime'].dt.month.isin([6,7,8])].mean()),
                        "southern": float(south_data[data['datetime'].dt.month.isin([6,7,8])].mean())
                    },
                    "winter": {
                        "northern": float(north_data[data['datetime'].dt.month.isin([12,1,2])].mean()),
                        "central": float(central_data[data['datetime'].dt.month.isin([12,1,2])].mean()),
                        "southern": float(south_data[data['datetime'].dt.month.isin([12,1,2])].mean())
                    }
                },
                "latitude_effect": {
                    "north_south_difference": float(south_data.mean() - north_data.mean()),
                    "relative_efficiency": float(south_data.mean() / north_data.mean())
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing geographical patterns: {str(e)}")

    def analyze_clear_sky_patterns(self, data: pd.DataFrame, country_code: str = None):
        """Analyze patterns suggesting clear sky vs cloudy conditions."""
        try:
            data['datetime'] = pd.to_datetime(data['time'])
            if country_code:
                if country_code not in data.columns:
                    raise ValueError(f"No data available for country code: {country_code}")
                solar_data = data[country_code]
            else:
                solar_data = data.drop(['time', 'datetime'], axis=1).mean(axis=1)
            
            # Calculate daily maximum solar radiation
            daily_max = solar_data.groupby(data['datetime'].dt.date).max()
            
            # Define clear sky threshold (90th percentile of daily maximums)
            clear_sky_threshold = daily_max.quantile(0.9)
            
            # Identify clear sky periods
            clear_sky_days = daily_max[daily_max >= clear_sky_threshold]
            
            # Convert index to datetime series for analysis
            clear_sky_dates = pd.Series(clear_sky_days.index)
            
            # Calculate monthly and seasonal distributions
            monthly_dist = {}
            for month in range(1, 13):
                monthly_count = sum(pd.to_datetime(date).month == month for date in clear_sky_dates)
                monthly_dist[str(month)] = int(monthly_count)
            
            # Calculate seasonal distributions
            summer_days = sum(pd.to_datetime(date).month in [6,7,8] for date in clear_sky_dates)
            winter_days = sum(pd.to_datetime(date).month in [12,1,2] for date in clear_sky_dates)
            
            return {
                "analysis": "clear_sky_patterns",
                "clear_sky_metrics": {
                    "clear_sky_threshold": float(clear_sky_threshold),
                    "clear_sky_days_percentage": float(len(clear_sky_days) / len(daily_max)),
                    "average_clear_sky_intensity": float(clear_sky_days.mean())
                },
                "monthly_distribution": monthly_dist,
                "seasonal_patterns": {
                    "summer_clear_days": float(summer_days / len(clear_sky_dates)) if len(clear_sky_dates) > 0 else 0.0,
                    "winter_clear_days": float(winter_days / len(clear_sky_dates)) if len(clear_sky_dates) > 0 else 0.0
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing clear sky patterns: {str(e)}")
        
    def analyze_country(self, data: pd.DataFrame, country_code: str):
        """Analyze solar patterns for a specific country."""
        try:
            if country_code not in data.columns:
                raise ValueError(f"No data available for country code: {country_code}")
            
            data['datetime'] = pd.to_datetime(data['time'])
            country_data = data[country_code]
            
            # Calculate seasonal averages
            seasonal_data = {
                "winter": float(country_data[data['datetime'].dt.month.isin([12,1,2])].mean()),
                "spring": float(country_data[data['datetime'].dt.month.isin([3,4,5])].mean()),
                "summer": float(country_data[data['datetime'].dt.month.isin([6,7,8])].mean()),
                "fall": float(country_data[data['datetime'].dt.month.isin([9,10,11])].mean())
            }
            
            return {
                "analysis": "country_analysis",
                "country_code": country_code,
                "overall_metrics": {
                    "average_solar_output": float(country_data.mean()),
                    "maximum_output": float(country_data.max()),
                    "minimum_output": float(country_data.min()),
                    "output_variability": float(country_data.std())
                },
                "seasonal_patterns": seasonal_data,
                "daily_cycle": {
                    str(hour): float(country_data[data['datetime'].dt.hour == hour].mean())
                    for hour in range(24)
                },
                "optimal_generation_hours": {
                    "start_hour": int(country_data.groupby(data['datetime'].dt.hour).mean().nlargest(8).index[0]),
                    "peak_hour": int(country_data.groupby(data['datetime'].dt.hour).mean().idxmax()),
                    "end_hour": int(country_data.groupby(data['datetime'].dt.hour).mean().nlargest(8).index[-1])
                }
            }
        except Exception as e:
            raise Exception(f"Error in country analysis: {str(e)}")

    def compare_regions(self, data: pd.DataFrame, country1: str, country2: str):
        """Compare solar potential between two countries."""
        try:
            if country1 not in data.columns:
                raise ValueError(f"No data available for country code: {country1}")
            if country2 not in data.columns:
                raise ValueError(f"No data available for country code: {country2}")
            
            data['datetime'] = pd.to_datetime(data['time'])
            
            # Calculate metrics for both countries
            metrics1 = self._calculate_country_metrics(data, country1)
            metrics2 = self._calculate_country_metrics(data, country2)
            
            return {
                "analysis": "regional_comparison",
                "country1": {
                    "code": country1,
                    **metrics1
                },
                "country2": {
                    "code": country2,
                    **metrics2
                },
                "comparison": {
                    "average_difference": float(metrics1["average_output"] - metrics2["average_output"]),
                    "daylight_hours_difference": float(metrics1["daylight_hours"] - metrics2["daylight_hours"]),
                    "relative_efficiency": float(metrics1["average_output"] / metrics2["average_output"])
                },
                "seasonal_comparison": {
                    season: {
                        "country1": float(metrics1["seasonal_output"][season]),
                        "country2": float(metrics2["seasonal_output"][season]),
                        "difference": float(metrics1["seasonal_output"][season] - metrics2["seasonal_output"][season])
                    }
                    for season in ["winter", "spring", "summer", "fall"]
                }
            }
        except Exception as e:
            raise Exception(f"Error comparing regions: {str(e)}")

    def analyze_seasonal_efficiency(self, data: pd.DataFrame, country_code: str = None):
        """Analyze seasonal solar efficiency patterns."""
        try:
            data['datetime'] = pd.to_datetime(data['time'])
            
            if country_code:
                if country_code not in data.columns:
                    raise ValueError(f"No data available for country code: {country_code}")
                solar_data = data[country_code]
            else:
                solar_data = data.drop(['time', 'datetime'], axis=1).mean(axis=1)
            
            # Calculate seasonal metrics
            seasonal_data = {
                "winter": solar_data[data['datetime'].dt.month.isin([12,1,2])],
                "spring": solar_data[data['datetime'].dt.month.isin([3,4,5])],
                "summer": solar_data[data['datetime'].dt.month.isin([6,7,8])],
                "fall": solar_data[data['datetime'].dt.month.isin([9,10,11])]
            }
            
            return {
                "analysis": "seasonal_efficiency",
                "seasonal_metrics": {
                    season: {
                        "average_output": float(data.mean()),
                        "peak_output": float(data.max()),
                        "output_stability": float(data.std() / data.mean()),  # Coefficient of variation
                        "daylight_hours": float(len(data[data > 0]) / len(data) * 24)
                    }
                    for season, data in seasonal_data.items()
                },
                "seasonal_comparisons": {
                    "best_season": max(seasonal_data.items(), key=lambda x: x[1].mean())[0],
                    "worst_season": min(seasonal_data.items(), key=lambda x: x[1].mean())[0],
                    "seasonal_variation": float(max(d.mean() for d in seasonal_data.values()) - 
                                              min(d.mean() for d in seasonal_data.values()))
                },
                "monthly_progression": {
                    str(month): float(solar_data[data['datetime'].dt.month == month].mean())
                    for month in range(1, 13)
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing seasonal efficiency: {str(e)}")

    def _calculate_country_metrics(self, data: pd.DataFrame, country_code: str) -> dict:
        """Helper function to calculate comprehensive metrics for a country."""
        country_data = data[country_code]
        
        seasonal_output = {
            "winter": float(country_data[data['datetime'].dt.month.isin([12,1,2])].mean()),
            "spring": float(country_data[data['datetime'].dt.month.isin([3,4,5])].mean()),
            "summer": float(country_data[data['datetime'].dt.month.isin([6,7,8])].mean()),
            "fall": float(country_data[data['datetime'].dt.month.isin([9,10,11])].mean())
        }
        
        return {
            "average_output": float(country_data.mean()),
            "peak_output": float(country_data.max()),
            "daylight_hours": float(len(country_data[country_data > 0]) / len(country_data) * 24),
            "output_stability": float(country_data.std() / country_data.mean()),
            "seasonal_output": seasonal_output
        }