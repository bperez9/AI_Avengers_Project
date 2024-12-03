# main.py

import json
import pandas as pd
from Tools.sea_level_tool import SeaLevelTool
from groq import Groq  # Replace with your actual client import
from Configurations.api import API_KEY

# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'  # Replace with your actual model name
client = Groq(api_key=API_KEY)

# Load your sea level and GMSL data into DataFrames
sea_level_df = pd.read_csv('Datasets/Sea_Level_cleaned 2.csv')
gsml_df = pd.read_csv('Datasets/GSML_cleaned 2.csv')

# Initialize the tool with the sea level and GMSL data
sea_level_tool = SeaLevelTool(sea_level_df, gsml_df)

# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": sea_level_tool.get_function_definition()
    },
]

# Map function names to functions
available_functions = {
    sea_level_tool.get_name(): sea_level_tool.run_impl,
}


# Function to handle the LLM conversation
def run_conversation():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sea level and climate change assistant for the year 2024. Given information about sea levels "
                "and global mean sea level (GMSL), you must always call the appropriate function to analyze the data "
                "using the provided information. Do not provide answers without using the functions. "
                "If the user does not specify specific years or parameters, you should ask for clarification. "
                "You are not going to answer questions about anything else, you will only answer questions that are "
                "related to sea level rise and global mean sea level."
            )
        },
    ]

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the conversation.")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        tool_calls = response_message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                function_response = function_to_call(**function_args)

                tool_response_message = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
                messages.append(tool_response_message)

            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            second_response_message = second_response.choices[0].message
            messages.append(second_response_message)

            print("Assistant:", second_response_message.content)
        else:
            print("Assistant:", response_message.content)

if __name__ == "__main__":
    print("Start chatting with the sea level assistant (type 'exit' or 'quit' to stop):")
    run_conversation()