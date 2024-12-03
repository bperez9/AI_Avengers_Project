from typing import Dict, Optional, List, Tuple
from Base_Tool.base_tool import SingleMessageTool
import pandas as pd
import numpy as np
from math import atan2, degrees

class TornadoAnalysisTool(SingleMessageTool):
    """Tool for analyzing tornado data from Tornados.csv"""

    def __init__(self, tornado_data):
        """Initialize with tornado data."""
        self.tornado_data = tornado_data

    def get_name(self) -> str:
        return "analyze_tornado_data"

    def get_description(self) -> str:
        return ("Analyze tornado data to provide insights about patterns, impacts, "
                "and characteristics across different dimensions.")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "analysis_type": {
                "description": ("Type of analysis: 'severity_impact', 'path_characteristics', "
                              "'temporal_patterns', 'state_comparison', 'economic_impact', "
                              "'movement_analysis', 'f_scale_distribution'"),
                "type": "string",
                "required": True,
                "enum": [
                    "severity_impact", "path_characteristics", "temporal_patterns",
                    "state_comparison", "economic_impact", "movement_analysis",
                    "f_scale_distribution"
                ]
            },
            "state": {
                "description": "State code for analysis (e.g., 'TX', 'OK')",
                "type": "string",
                "required": False
            },
            "comparison_state": {
                "description": "Second state code for comparison",
                "type": "string",
                "required": False
            },
            "year": {
                "description": "Year for specific analysis",
                "type": "integer",
                "required": False
            },
            "min_length": {
                "description": "Minimum tornado path length for filtering",
                "type": "number",
                "required": False
            }
        }

    def run_impl(self, analysis_type: str, state: str = None, comparison_state: str = None, 
                 year: int = None, min_length: float = None):
        """Implement the tool logic."""
        try:
            if analysis_type == "severity_impact":
                return self.analyze_severity_impact(year, state)
            elif analysis_type == "path_characteristics":
                return self.analyze_path_characteristics(state, min_length)
            elif analysis_type == "temporal_patterns":
                return self.analyze_temporal_patterns(state)
            elif analysis_type == "state_comparison":
                if not state:
                    raise ValueError("State code is required for state comparison")
                return self.compare_states(state, comparison_state)
            elif analysis_type == "economic_impact":
                return self.analyze_economic_impact(state, year)
            elif analysis_type == "movement_analysis":
                return self.analyze_tornado_movement(min_length)
            elif analysis_type == "f_scale_distribution":
                return self.analyze_f_scale_distribution(state)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            raise Exception(f"Error in tornado analysis: {str(e)}")

    def analyze_severity_impact(self, year: int = None, state: str = None):
        """Analyze relationship between magnitude and impact."""
        try:
            data = self.tornado_data
            if year:
                data = data[data['yr'] == year]
            if state:
                data = data[data['st'] == state]
            
            return {
                "analysis": "severity_impact",
                "magnitude_distribution": {
                    str(mag): {
                        "count": int(count),
                        "avg_injuries": float(data[data['mag'] == mag]['inj'].mean()),
                        "avg_fatalities": float(data[data['mag'] == mag]['fat'].mean()),
                        "avg_loss": float(data[data['mag'] == mag]['loss'].mean())
                    }
                    for mag, count in data['mag'].value_counts().items()
                },
                "correlation": {
                    "magnitude_injuries": float(data['mag'].corr(data['inj'])),
                    "magnitude_fatalities": float(data['mag'].corr(data['fat'])),
                    "magnitude_loss": float(data['mag'].corr(data['loss']))
                },
                "total_impact": {
                    "total_injuries": int(data['inj'].sum()),
                    "total_fatalities": int(data['fat'].sum()),
                    "total_loss": float(data['loss'].sum())
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing severity impact: {str(e)}")

    def _calculate_path_directions(self, data: pd.DataFrame) -> Dict:
        """Calculate tornado path directions using start/end coordinates."""
        directions = []
        for _, row in data.iterrows():
            if row['elat'] != 0 and row['elon'] != 0:  # Check for valid end coordinates
                dy = row['elat'] - row['slat']
                dx = row['elon'] - row['slon']
                angle = degrees(atan2(dy, dx))
                directions.append(angle)
        
        if directions:
            return {
                "avg_direction": float(np.mean(directions)),
                "direction_distribution": {
                    "north": len([d for d in directions if 45 <= d < 135]),
                    "south": len([d for d in directions if -135 <= d < -45]),
                    "east": len([d for d in directions if (-45 <= d < 45)]),
                    "west": len([d for d in directions if (d >= 135) or (d < -135)])
                }
            }
        return {"error": "No valid direction data available"}

    def analyze_path_characteristics(self, state: str = None, min_length: float = None):
        """Analyze tornado paths (length, width, direction)."""
        try:
            data = self.tornado_data
            if state:
                data = data[data['st'] == state]
            if min_length:
                data = data[data['len'] >= min_length]
            
            return {
                "analysis": "path_characteristics",
                "path_metrics": {
                    "avg_length": float(data['len'].mean()),
                    "avg_width": float(data['wid'].mean()),
                    "max_length": float(data['len'].max()),
                    "max_width": float(data['wid'].max())
                },
                "size_distribution": {
                    "length_quartiles": {
                        str(q): float(val) 
                        for q, val in data['len'].quantile([0.25, 0.5, 0.75]).items()
                    },
                    "width_quartiles": {
                        str(q): float(val)
                        for q, val in data['wid'].quantile([0.25, 0.5, 0.75]).items()
                    }
                },
                "direction_analysis": self._calculate_path_directions(data),
                "path_count": len(data)
            }
        except Exception as e:
            raise Exception(f"Error analyzing path characteristics: {str(e)}")

    def _analyze_time_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze distribution of tornadoes across different times."""
        time_counts = data['time'].value_counts()
        return {
            "morning": int(time_counts[(time_counts.index >= '06:00:00') & 
                                     (time_counts.index < '12:00:00')].sum()),
            "afternoon": int(time_counts[(time_counts.index >= '12:00:00') & 
                                       (time_counts.index < '18:00:00')].sum()),
            "evening": int(time_counts[(time_counts.index >= '18:00:00') & 
                                     (time_counts.index < '22:00:00')].sum()),
            "night": int(time_counts[(time_counts.index >= '22:00:00') | 
                                   (time_counts.index < '06:00:00')].sum())
        }

    def analyze_temporal_patterns(self, state: str = None):
        """Analyze patterns across different time scales."""
        try:
            data = self.tornado_data
            if state:
                data = data[data['st'] == state]
            
            return {
                "analysis": "temporal_patterns",
                "yearly_trends": {
                    str(year): {
                        "count": int(count),
                        "avg_magnitude": float(data[data['yr'] == year]['mag'].mean()),
                        "total_injuries": int(data[data['yr'] == year]['inj'].sum()),
                        "total_fatalities": int(data[data['yr'] == year]['fat'].sum())
                    }
                    for year, count in data['yr'].value_counts().items()
                },
                "monthly_distribution": {
                    str(month): int(count)
                    for month, count in data['mo'].value_counts().sort_index().items()
                },
                "time_of_day": self._analyze_time_distribution(data),
                "seasonal_patterns": {
                    "spring": int(data[data['mo'].isin([3,4,5])].shape[0]),
                    "summer": int(data[data['mo'].isin([6,7,8])].shape[0]),
                    "fall": int(data[data['mo'].isin([9,10,11])].shape[0]),
                    "winter": int(data[data['mo'].isin([12,1,2])].shape[0])
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing temporal patterns: {str(e)}") 

    def _calculate_state_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics for a state."""
        return {
            "total_tornadoes": len(data),
            "avg_magnitude": float(data['mag'].mean()),
            "total_injuries": int(data['inj'].sum()),
            "total_fatalities": int(data['fat'].sum()),
            "total_loss": float(data['loss'].sum()),
            "avg_path_length": float(data['len'].mean()),
            "avg_path_width": float(data['wid'].mean())
        }

    def compare_states(self, state1: str, state2: str = None):
        """Compare tornado characteristics between states."""
        try:
            data = self.tornado_data
            state1_data = data[data['st'] == state1]
            
            if not len(state1_data):
                raise ValueError(f"No data available for state: {state1}")
            
            metrics1 = self._calculate_state_metrics(state1_data)
            
            if state2:
                state2_data = data[data['st'] == state2]
                if not len(state2_data):
                    raise ValueError(f"No data available for state: {state2}")
                
                metrics2 = self._calculate_state_metrics(state2_data)
                return {
                    "analysis": "state_comparison",
                    "state1": {
                        "state_code": state1,
                        **metrics1
                    },
                    "state2": {
                        "state_code": state2,
                        **metrics2
                    },
                    "differences": {
                        key: float(metrics1[key] - metrics2[key])
                        for key in metrics1.keys()
                        if isinstance(metrics1[key], (int, float))
                    }
                }
            else:
                # Compare with national averages
                national_metrics = self._calculate_state_metrics(data)
                return {
                    "analysis": "state_comparison",
                    "state": {
                        "state_code": state1,
                        **metrics1
                    },
                    "national_average": national_metrics,
                    "relative_metrics": {
                        key: float(metrics1[key] / national_metrics[key])
                        for key in metrics1.keys()
                        if isinstance(metrics1[key], (int, float)) and national_metrics[key] != 0
                    }
                }
        except Exception as e:
            raise Exception(f"Error comparing states: {str(e)}")

    def analyze_economic_impact(self, state: str = None, year_range: tuple = None):
        """Analyze economic losses and patterns."""
        try:
            data = self.tornado_data
            if state:
                data = data[data['st'] == state]
            if year_range:
                data = data[(data['yr'] >= year_range[0]) & (data['yr'] <= year_range[1])]
            
            yearly_losses = data.groupby('yr')['loss'].agg(['sum', 'mean', 'count'])
            
            return {
                "analysis": "economic_impact",
                "total_loss": float(data['loss'].sum()),
                "avg_loss_per_tornado": float(data['loss'].mean()),
                "loss_by_magnitude": {
                    str(mag): float(avg_loss)
                    for mag, avg_loss in data.groupby('mag')['loss'].mean().items()
                },
                "yearly_trends": {
                    str(year): {
                        "total_loss": float(row['sum']),
                        "avg_loss": float(row['mean']),
                        "tornado_count": int(row['count'])
                    }
                    for year, row in yearly_losses.iterrows()
                },
                "correlation_metrics": {
                    "loss_vs_magnitude": float(data['loss'].corr(data['mag'])),
                    "loss_vs_path_length": float(data['loss'].corr(data['len'])),
                    "loss_vs_path_width": float(data['loss'].corr(data['wid']))
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing economic impact: {str(e)}")

    def analyze_tornado_movement(self, min_length: float = None):
        """Analyze tornado movement patterns using start/end coordinates."""
        try:
            data = self.tornado_data
            if min_length:
                data = data[data['len'] >= min_length]
            
            # Filter for valid coordinates
            valid_data = data[(data['elat'] != 0) & (data['elon'] != 0)]
            
            return {
                "analysis": "movement_analysis",
                "path_statistics": {
                    "avg_length": float(valid_data['len'].mean()),
                    "max_length": float(valid_data['len'].max()),
                    "length_distribution": {
                        str(q): float(val)
                        for q, val in valid_data['len'].quantile([0.25, 0.5, 0.75]).items()
                    }
                },
                "direction_analysis": self._calculate_path_directions(valid_data),
                "movement_patterns": {
                    "avg_displacement": float(
                        np.mean(np.sqrt(
                            (valid_data['elat'] - valid_data['slat'])**2 + 
                            (valid_data['elon'] - valid_data['slon'])**2
                        ))
                    ),
                    "path_efficiency": float(
                        np.mean(np.sqrt(
                            (valid_data['elat'] - valid_data['slat'])**2 + 
                            (valid_data['elon'] - valid_data['slon'])**2
                        ) / valid_data['len'])
                    )
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing tornado movement: {str(e)}")

    def analyze_f_scale_distribution(self, state: str = None):
        """Analyze F-scale ratings distribution and their relationships."""
        try:
            data = self.tornado_data
            if state:
                data = data[data['st'] == state]
            
            f_scale_cols = ['f1', 'f2', 'f3', 'f4']
            f_scale_data = data[f_scale_cols]
            
            return {
                "analysis": "f_scale_distribution",
                "f_scale_counts": {
                    f"F{i+1}": int(f_scale_data[col].sum())
                    for i, col in enumerate(f_scale_cols)
                },
                "f_scale_correlations": {
                    "magnitude": {
                        f"F{i+1}": float(data['mag'].corr(data[col]))
                        for i, col in enumerate(f_scale_cols)
                    },
                    "damage": {
                        f"F{i+1}": float(data['loss'].corr(data[col]))
                        for i, col in enumerate(f_scale_cols)
                    }
                },
                "temporal_distribution": {
                    str(year): {
                        f"F{i+1}": int(year_data[col].sum())
                        for i, col in enumerate(f_scale_cols)
                    }
                    for year, year_data in data.groupby('yr')[f_scale_cols]
                }
            }
        except Exception as e:
            raise Exception(f"Error analyzing F-scale distribution: {str(e)}")