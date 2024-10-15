# main.py

import json
import pandas as pd
from tools.extreme_weather_tool import ExtremeWeatherTool
from groq import Groq  # Replace with your actual client import

# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'# Replace with your actual model name
API_KEY = ""   # Replace with your actual API key

# Load your weather data into a DataFrame
weather_data = pd.read_csv('data/weather-statistics-hourly.csv')

# Initialize the tool with the weather data
extreme_weather_tool = ExtremeWeatherTool(weather_data)

# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": extreme_weather_tool.get_function_definition()
    },
    # Add other tools here if needed
]

# Map function names to functions
available_functions = {
    extreme_weather_tool.get_name(): extreme_weather_tool.run_impl,
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
                "If the user does not specify the event type or threshold, you should assume default values: "
                "'heatwave' for event type and 35 degrees Celsius for threshold."
                "You are not going to answer questions about anything else, you will only answer questions that are climate related"
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

