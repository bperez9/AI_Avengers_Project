import json
import pandas as pd
from Tools.fuel_efficiency_tool import FuelConsumptionTool
from groq import Groq  # Replace with your actual client import
from Configurations.api import API_KEY

# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'  # Replace with your actual model name
client = Groq(api_key=API_KEY)  # Replace with your actual API key

# Load your fuel consumption data into a DataFrame
fuel_data = pd.read_csv('Datasets/fuel_consumption_canada.csv')

# Initialize the tool with the fuel data
fuel_consumption_tool = FuelConsumptionTool(fuel_data)

# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "average_co2_by_make",
            "params": {},  # Specify any required parameters if applicable
        }
    },
    {
        "type": "function",
        "function": {
            "name": "highest_co2_emissions",
            "params": {},  # Specify any required parameters if applicable
        }
    },
    {
        "type": "function",
        "function": {
            "name": "average_engine_size_by_vehicle_class",
            "params": {},  # Specify any required parameters if applicable
        }
    },
    {
        "type": "function",
        "function": {
            "name": "average_fuel_efficiency_by_class",  # Corrected name
            "params": {},  # Specify any required parameters if applicable
        }
    },
]



# Map function names to functions
available_functions = {
    fuel_consumption_tool.get_name(): fuel_consumption_tool.run_impl,
    "average_co2_by_make": fuel_consumption_tool.average_co2_emissions_by_make,
    "highest_co2_emissions": fuel_consumption_tool.cars_with_highest_co2_emissions,
    "average_engine_size_by_vehicle_class": fuel_consumption_tool.average_engine_size_by_vehicle_class,
    "average_fuel_efficiency_by_class": fuel_consumption_tool.average_combined_fuel_efficiency_by_vehicle_class,  # Correct method name
}



# Function to handle the LLM conversation
def run_conversation():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fuel efficiency assistant for Canada. Given information about fuel consumption and emissions, "
                "you must always call the appropriate function to identify fuel-efficient cars using the provided data. "
                "If the user does not specify the year or fuel type, you should assume default values: "
                "'All' for fuel type and the latest year in the dataset. "
                "You are not going to answer questions about anything else, you will only answer questions that are related to fuel consumption."
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
