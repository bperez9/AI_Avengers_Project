# main.py
import json
import pandas as pd
from tools.surfaces_tool import GetTotalArtificialSurfaces, GetTotalShrubCoveredAreas
from tools.extreme_weather_tool import ExtremeWeatherTool
from tools.fuel_tool import FuelConsumptionTool
from tools.temp_analysis_tool import TemperatureAnalysisTool
from tools.weather_query_tool import PandasAITool
from groq import Groq  
from dotenv import load_dotenv
import os

load_dotenv()

# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'
API_KEY = os.getenv('GROQ_API')
# API_KEY = ""   # Replace with your actual API key

# Load your weather data into a DataFrame
weather_data = pd.read_csv('data/weather-statistics-hourly.csv')
fuel_data = pd.read_csv('data/fuel_consumption_canada.csv')
fuel_data.rename(columns={'Model year': 'year'}, inplace=True)
city_data = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')
country_data = pd.read_csv('data/GlobalLandTemperaturesByCountry.csv')
global_data = pd.read_csv('data/GlobalTemperatures.csv')

# Initialize the tool with the weather data
total_artificial_surfaces_tool = GetTotalArtificialSurfaces()
total_shrub_covered_areas_tool = GetTotalShrubCoveredAreas()
extreme_weather_tool = ExtremeWeatherTool(weather_data)
fuel_consumption_tool = FuelConsumptionTool(fuel_data)
temperature_analysis_tool = TemperatureAnalysisTool(city_data, country_data, global_data)
query_weather_data_tool = PandasAITool(weather_data)

# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": extreme_weather_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": total_artificial_surfaces_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": total_shrub_covered_areas_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": fuel_consumption_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": temperature_analysis_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": query_weather_data_tool.get_function_definition()
    },
    # Add other tools here if needed
]

# Map function names to functions
available_functions = {
    total_artificial_surfaces_tool.get_name(): total_artificial_surfaces_tool.run_impl,
    total_shrub_covered_areas_tool.get_name(): total_shrub_covered_areas_tool.run_impl,
    extreme_weather_tool.get_name(): extreme_weather_tool.run_impl,
    fuel_consumption_tool.get_name(): fuel_consumption_tool.run_impl,
    temperature_analysis_tool.get_name(): temperature_analysis_tool.run_impl,
    query_weather_data_tool.get_name(): query_weather_data_tool.run_impl,
    # Add other function mappings here if needed
}

# Initialize your client for the LLM API
client = Groq(api_key=API_KEY)

# Function to handle the LLM conversation
def run_conversation():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a weather assistant for the year 2024. Given information about weather events, "
                "you must always call the appropriate function to identify regions experiencing extreme weather "
                "using the provided data. Do not provide answers without using the functions. "
                "When presenting the results, you should summarize the findings, provide context, and format the information "
                "in a way that is easy to understand for a normal person. Avoid overwhelming the user with long lists; instead, "
                "highlight key points and provide examples. "
                "- `query_weather_data`: Answer questions by analyzing the weather dataset."
                "If the user does not specify the event type or threshold, you should use default values and provide "
                "information about both 'heatwave' (threshold 35 degrees Celsius) and 'heavy_rainfall' (threshold 50 millimeters). "
                "You should not ask the user for more information, but proceed with the defaults if necessary. "
                "You are not going to answer questions about anything else; you will only answer questions that are climate-related. "
                "If the user asks a question that is not climate-related, politely inform them that you can only assist with climate-related questions. "
                "For every response, you must provide sources to verify the information."
                "Summarize key points, provide context, and avoid simply listing raw data unless necessary. "
                "If the function call yields no output, I want you to answer the questions based on your knwoledge and mention that the data did not return anything and hence you are giving the knowledge to your ability."
            )
        },
    ]


    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the conversation.")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        # Call the LLM API to get the assistant's response
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )

        response_message = response.choices[0].message
        messages.append(response_message)  # Append the assistant's response to the messages
        # Print the messages for debugging
        # print("\n[DEBUG] Conversation Messages:")
        # for msg in messages:
        #     print(msg)

        tool_calls = response_message.tool_calls
        print("\n[DEBUG] Tool Calls:")
        print(tool_calls)


        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                # Call the function with the provided arguments
                function_response = function_to_call(**function_args)

                # Serialize the function response to a JSON string
                tool_response_message = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
                messages.append(tool_response_message)

            # Send the updated conversation with tool response back to the model
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            second_response_message = second_response.choices[0].message
            messages.append(second_response_message)

            print("Assistant:", second_response_message.content)
        else:
            # No tool call; just print the assistant's response
            print("Assistant:", response_message.content)


if __name__ == "__main__":
    print("Start chatting with the assistant (type 'exit' or 'quit' to stop):")
    run_conversation()

