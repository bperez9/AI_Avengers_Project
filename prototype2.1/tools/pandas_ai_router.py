from typing import Dict
from Base_Tool.base_tool import SingleMessageTool
from pandasai import SmartDataframe
from dotenv import load_dotenv
import os

load_dotenv()

PANDASAI_API_KEY = os.getenv('OPENAI_API_KEY')

class PandasAIRouter(SingleMessageTool):
    """General purpose tool that can answer detailed questions about any dataset using PandasAI."""

    def __init__(self, datasets_dict):
        """
        Initialize with a dictionary of datasets.
        
        Args:
            datasets_dict: Dictionary mapping dataset names to their pandas DataFrames
                         e.g., {"weather_data": weather_df, "fuel_data": fuel_df}
        """
        self.datasets = datasets_dict
        self.dataset_descriptions = {
            "weather_data": "Hourly weather statistics dataset containing measurements of temperature, humidity, precipitation, wind, pressure, visibility, UV index, and soil conditions for different countries.",
            "fuel_data": "Dataset containing fuel consumption information for vehicles in Canada, including model year, make, model, and various fuel consumption metrics.",
            "city_data": "Dataset containing temperature data for various cities globally over time.",
            "country_data": "Dataset containing temperature data aggregated by country over time.",
            "global_data": "Dataset containing global temperature measurements and averages over time."
            # Add descriptions for other datasets as needed
        }

    def get_name(self) -> str:
        return "query_dataset"

    def get_description(self) -> str:
        return (
            "Use this tool for analyzing any dataset when the query doesn't fit the predefined functions. "
            "Available datasets:\n" + 
            "\n".join(f"- {name}: {desc}" for name, desc in self.dataset_descriptions.items())
        )

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "dataset_name": {
                "description": "Name of the dataset to query: " + ", ".join(self.datasets.keys()),
                "type": "string",
                "required": True
            },
            "question": {
                "description": "The question to analyze about the specified dataset",
                "type": "string",
                "required": True
            }
        }

    def run_impl(self, dataset_name: str, question: str):
        try:
            if dataset_name not in self.datasets:
                return {
                    "error": f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(self.datasets.keys())}",
                    "note": "Please specify a valid dataset name"
                }

            # Append instruction for text-only response
            enhanced_question = (
                f"{question} "
                "Provide the answer in detailed text format only, without any graphs or visualizations. "
                "Include specific numbers and statistics where relevant."
            )

            df = SmartDataframe(
                self.datasets[dataset_name],
                name=dataset_name,
                description=self.dataset_descriptions.get(dataset_name, "Dataset for analysis")
            )
            
            response = df.chat(enhanced_question)
            return {
                "answer": response,
                "note": f"Analysis based on {dataset_name}"
            }
        except Exception as e:
            return {
                "error": str(e),
                "note": f"Failed to analyze {dataset_name}"
            } 