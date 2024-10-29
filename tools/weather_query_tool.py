# tools/pandas_ai_tool.py

from typing import Dict
from tools.base_tool import SingleMessageTool
from pandasai import SmartDataframe, SmartDatalake
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

PANDASAI_API_KEY = os.getenv('OPENAI_API_KEY') 

weather_data = pd.read_csv('data/weather-statistics-hourly.csv')

class PandasAITool(SingleMessageTool):
    """Tool that can answer questions on the weather data which do not need functions"""

    def __init__(self, weather_data):
        self.weather_data = weather_data

    def get_name(self) -> str:
        return "query_weather_data"

    def get_description(self) -> str:
        return ("Use this tool to answer any questions about weather data, including questions about "
                "temperature, humidity, precipitation, and other weather metrics. This tool can analyze "
                "patterns, calculate averages, find extremes, and provide statistics from the weather dataset. "
                "Examples: 'What is the average humidity?', 'What are the temperature trends?', "
                "'How much rainfall was there?'")

    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "question": {
                "description": "The weather-related question to analyze. Can ask about any weather metric, patterns, or statistics.",
                "type": "string",
                "required": True,
                "examples": ["What is the average humidity?", "What are the highest temperatures recorded?"]
            },
        }

    def run_impl(self, question: str):
        try:
            # print(f"[DEBUG] Running PandasAITool with question: {question}")
            # Use the chat method of SmartDataframe to answer the question
                    # Initialize SmartDataframe with the weather data
            df = SmartDataframe(
                self.weather_data,
                name="Weather Data",
                description="Contains weather statistics including temperature, precipitation, humidity, soil temperature and etc."
            )
            response = df.chat(question)
            # print(f"[DEBUG] Received response from PandasAI: {response}")
            return {"answer": response}
        except Exception as e:
            # print(f"[ERROR] Exception in PandasAITool: {e}")
            return {"error": str(e)}
