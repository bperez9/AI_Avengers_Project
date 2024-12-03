# main.py

import json
import pandas as pd
from Tools.temp_analysis_tool import TemperatureAnalysisTool
from groq import Groq  # Replace with your actual client import
from Configurations.api import API_KEY

# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'  # Replace with your actual model name
client = Groq(api_key=API_KEY)  # Replace with your actual API key

# Load your temperature data into DataFrames
city_data = pd.read_csv('Datasets/GlobalLandTemperaturesByCity.csv')  

# Initialize the temperature analysis tool with the loaded data
temperature_analysis_tool = TemperatureAnalysisTool(city_data)

# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": temperature_analysis_tool.get_function_definition()
    },
]

# Map function names to functions
available_functions = {
    temperature_analysis_tool.get_name(): temperature_analysis_tool.run_impl,
}

# Function to handle the LLM conversation
def run_conversation():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a temperature analysis assistant. You must respond to user queries "
                "about temperature data, providing insights based on city or country temperatures. Use the provided "
                "data to analyze average temperatures, trends, and comparisons. Do not provide answers without using the functions."
                "Feel free to ask clarifying questions if the user input is incomplete or unclear."
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

        tool_calls = response_message.tool_calls

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
