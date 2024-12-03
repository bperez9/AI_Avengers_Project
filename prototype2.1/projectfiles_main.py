# main.py
import json
import pandas as pd
from groq import Groq  
from Configurations.api import API_KEY
from Tools.pandas_ai_router import PandasAIRouter
from Tools.fuel_efficiency_tool import FuelConsumptionTool
from Tools.ghgcontribution_tool import GHGContributionTool
from Tools.ghgemission_tool import EmissionsTool
from Tools.sea_level_tool import SeaLevelTool
from Tools.temp_analysis_landocean_tool import LandTemperatureAnalysisTool
from Tools.temp_analysis_tool import TemperatureAnalysisTool
from Tools.extreme_weather_tool import ExtremeWeatherTool
from Tools.wind_national_tool import WindNationalTool
from Tools.on_offshore_wind_tool import OnOffshoreWindTool
from Tools.future_longterm_wind_tool import FutureLongtermWindTool
from Tools.tornado_analysis_tool import TornadoAnalysisTool
from Tools.solar_analysis_tool import SolarAnalysisTool
from Tools.landcover_tool import LandCoverTool


# Constants
MODEL = 'llama3-groq-70b-8192-tool-use-preview'
client = Groq(api_key=API_KEY)

# Load your weather data into a DataFrame
fuel_data = pd.read_csv('Datasets/fuel_consumption_canada.csv')
contribution_data = pd.read_csv('Datasets/CW_NDC.csv') 
emissions_df = pd.read_csv('Datasets/cleaned_essd_ghg_data.csv')
sea_level_df = pd.read_csv('Datasets/Sea_Level_cleaned 2.csv')
gsml_df = pd.read_csv('Datasets/GSML_cleaned 2.csv')
land_data = pd.read_csv('Datasets/GlobalLandOceanTemperatures.csv') 
city_data = pd.read_csv('Datasets/GlobalLandTemperaturesByCity.csv') 
weather_data = pd.read_csv('Datasets/weather-statistics-hourly.csv')
wind_national_data = pd.read_csv('Datasets/current_national.csv')
onoffshore_wind_data = pd.read_csv('Datasets/current_on_offshore.csv')
future_longterm_wind_data = pd.read_csv('Datasets/future_longterm_national.csv')
tornado_data = pd.read_csv('Datasets/Tornados.csv')
solar_sarah_data = pd.read_csv('Datasets/solar_sarah.csv')
solar_merra_data = pd.read_csv('Datasets/solar_merra2.csv')
land_cover_data = pd.read_csv('Datasets/Land_Cover_Accounts.csv')


# Load your datasets
datasets = {
    "fuel_data": fuel_data,
    "contribution_data":contribution_data,
    "emissions_df":emissions_df,
    "sea_level_df":sea_level_df,
    "gsml_df":gsml_df,
    "land_data":land_data,
    "city_data":city_data,
    "weather_data": weather_data,
    "wind_national_data": wind_national_data,
    "onoffshore_wind_data": onoffshore_wind_data,
    "future_longterm_wind_data": future_longterm_wind_data,
    "tornado_data": tornado_data,
    "solar_sarah_data": solar_sarah_data,
    "solar_merra_data": solar_merra_data,
    "land_cover_data":land_cover_data    
}

# Initialize the PandasAI router with all datasets
pandas_ai_router = PandasAIRouter(datasets)

# Initialize the tools
fuel_consumption_tool = FuelConsumptionTool(fuel_data)
ghg_contribution_tool = GHGContributionTool(contribution_data)
emissions_tool = EmissionsTool(emissions_df)
sea_level_tool = SeaLevelTool(sea_level_df, gsml_df)
land_temperature_analysis_tool = LandTemperatureAnalysisTool(land_data)
temperature_analysis_tool = TemperatureAnalysisTool(city_data)
extreme_weather_tool = ExtremeWeatherTool(weather_data)
wind_national_tool = WindNationalTool(wind_national_data)
onoffshore_wind_tool = OnOffshoreWindTool(onoffshore_wind_data)
future_longterm_wind_tool = FutureLongtermWindTool(future_longterm_wind_data)
tornado_analysis_tool = TornadoAnalysisTool(tornado_data)
solar_analysis_tool = SolarAnalysisTool(solar_sarah_data, solar_merra_data)
land_cover_tool = LandCoverTool(land_cover_data)


# Define the tools list for the LLM
tools = [
    {
        "type": "function",
        "function": pandas_ai_router.get_function_definition()
    },
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
    {
        "type": "function",
        "function": ghg_contribution_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": emissions_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": sea_level_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": land_temperature_analysis_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": temperature_analysis_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": extreme_weather_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": wind_national_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": onoffshore_wind_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": future_longterm_wind_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": tornado_analysis_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": solar_analysis_tool.get_function_definition()
    },
    {
        "type": "function",
        "function": land_cover_tool.get_function_definition()
    }
]

# Update the available_functions dictionary
available_functions = {
    pandas_ai_router.get_name(): pandas_ai_router.run_impl,
    fuel_consumption_tool.get_name(): fuel_consumption_tool.run_impl,
    "average_co2_by_make": fuel_consumption_tool.average_co2_emissions_by_make,
    "highest_co2_emissions": fuel_consumption_tool.cars_with_highest_co2_emissions,
    "average_engine_size_by_vehicle_class": fuel_consumption_tool.average_engine_size_by_vehicle_class,
    "average_fuel_efficiency_by_class": fuel_consumption_tool.average_combined_fuel_efficiency_by_vehicle_class,
    ghg_contribution_tool.get_name(): ghg_contribution_tool.run_impl,
    emissions_tool.get_name(): emissions_tool.run_impl,
    sea_level_tool.get_name(): sea_level_tool.run_impl,
    land_temperature_analysis_tool.get_name(): land_temperature_analysis_tool.run_impl,
    temperature_analysis_tool.get_name(): temperature_analysis_tool.run_impl,
    extreme_weather_tool.get_name(): extreme_weather_tool.run_impl,
    wind_national_tool.get_name(): wind_national_tool.run_impl,
    onoffshore_wind_tool.get_name(): onoffshore_wind_tool.run_impl,
    future_longterm_wind_tool.get_name(): future_longterm_wind_tool.run_impl,
    tornado_analysis_tool.get_name(): tornado_analysis_tool.run_impl,
    solar_analysis_tool.get_name(): solar_analysis_tool.run_impl,
    "analyze_land_cover_data": land_cover_tool.run_impl,    
}


# Function to handle the LLM conversation
def run_conversation():
    # Update the system message content to include wind national tool capabilities
    system_content = (
        "You are a data analysis assistant for the year 2024. You have access to several specialized tools and datasets. "
        "Your job is to route queries to the most appropriate tool or use PandasAI for complex analyses.\n\n"

        "1. SPECIALIZED TOOLS AND THEIR FUNCTIONS:\n\n"
        
        "a) Extreme Weather Tool (get_regions_with_extreme_weather):\n"
        "- Identifies regions with specific weather events\n"
        "- Event types: 'heatwave' (>35°C), 'heavy_rainfall' (>50mm), 'high_humidity' (>90%), 'strong_winds' (>20m/s), 'uv_warning' (>8), 'soil_analysis'\n"
        "- Example queries:\n"
        "  * 'Which regions have temperatures above 35°C?'\n"
        "  * 'Are there any areas with heavy rainfall?'\n\n"
        
        "b) Fuel Consumption Tool (get_most_fuel_efficient_cars):\n"
        "- Finds the most fuel-efficient vehicles for a given year\n"
        "- Parameters: year (required), fuel_type (optional)\n"
        "- Example queries:\n"
        "  * 'What was the most fuel-efficient car in 2010?'\n"
        "  * 'Show me the most efficient gasoline cars from 2015'\n\n"
        
        "c) Temperature Analysis Tool:\n"
        "- Analyzes temperature trends across cities and countries\n"
        "- Example queries:\n"
        "  * 'What's the temperature trend in Paris over time?'\n"
        "  * 'Compare temperatures between London and New York'\n\n"
        

        "e) Wind National Analysis Tool (analyze_national_wind_power):\n"
        "- Analyzes national wind power data and capacity factors\n"
        "- Provides three types of analysis:\n"
        "  1. Top Performers Analysis: Identifies countries with highest capacity factors\n"
        "  2. Seasonal Pattern Analysis: Shows wind power patterns across seasons\n"
        "  3. Country Comparison: Compares specific country's performance against others\n"
        "- Example queries:\n"
        "  * 'Show me the top 5 countries with highest wind power capacity'\n"
        "  * 'What are the seasonal wind power patterns in Europe?'\n"
        "  * 'Compare Germany's wind power performance with other countries'\n"
        "  * 'Show wind power patterns for summer season'\n"
        "  * 'Which countries have the best wind power performance?'\n\n"
        
        "f) Onshore/Offshore Wind Analysis Tool (analyze_onoffshore_wind_power):\n"
        "- Analyzes onshore and offshore wind power data separately\n"
        "- Provides four types of analysis:\n"
        "  1. Distribution Analysis: Shows distribution of onshore vs offshore installations\n"
        "  2. Efficiency Comparison: Compares efficiency between onshore and offshore\n"
        "  3. Top Producers: Lists top performing countries for each type\n"
        "  4. Country Detail: Detailed analysis for specific countries\n"
        "- Example queries:\n"
        "  * 'Compare efficiency between onshore and offshore wind power'\n"
        "  * 'Show me the top offshore wind power producers'\n"
        "  * 'What's the distribution of onshore vs offshore installations?'\n"
        "  * 'Give me details about Germany's onshore and offshore capacity'\n\n"

        "g) Future Long-term Wind Analysis Tool (analyze_future_longterm_wind):\n"
        "- Analyzes long-term future wind power projections\n"
        "- Provides four types of analysis:\n"
        "  1. Trend Analysis: Shows overall trends and top performing countries\n"
        "  2. Country Projection: Detailed future projections for specific countries\n"
        "  3. Comparative Analysis: Compares projections between countries\n"
        "  4. Peak Performance: Analyzes peak capacity periods\n"
        "- Example queries:\n"
        "  * 'What are the projected wind power trends for the future?'\n"
        "  * 'Show me Germany's future wind power projections'\n"
        "  * 'Compare future wind power between France and Spain'\n"
        "  * 'Which countries are expected to have peak performance?'\n\n"

        "h) Tornado Analysis Tool (analyze_tornado_data):\n"
        "- Analyzes historical tornado data and patterns\n"
        "- Provides multiple types of analysis:\n"
        "  1. Severity Impact: Analyzes relationship between magnitude and impacts\n"
        "  2. Path Characteristics: Analyzes tornado paths and movements\n"
        "  3. Temporal Patterns: Shows patterns across time scales\n"
        "  4. State Comparison: Compares tornado characteristics between states\n"
        "  5. Economic Impact: Analyzes loss patterns and trends\n"
        "  6. F-scale Distribution: Analyzes tornado intensity distributions\n"
        "- Example queries for tool functions:\n"
        "  * 'Show me the severity impact analysis for Texas'\n"
        "  * 'Compare tornado characteristics between Oklahoma and Kansas'\n"
        "  * 'What are the temporal patterns of tornadoes in Florida?'\n"
        "  * 'Analyze the economic impact of tornadoes in Illinois'\n"
        "  * 'Show me the F-scale distribution for all tornadoes'\n"

        "h) Solar Analysis Tool (analyze_solar_data):\n"
        "- Analyzes solar power data using two specialized datasets:\n"
        "  * SARAH: Specialized for solar energy applications\n"
        "  * MERRA: Broader environmental context\n"
        "- Provides six types of analysis:\n"
        "  1. Daylight Patterns: Analyzes daylight hours, sunrise/sunset times, and seasonal variations\n"
        "  2. Geographical Patterns: Compares Northern, Central, and Southern European regions\n"
        "  3. Clear Sky Patterns: Analyzes optimal solar conditions and their distribution\n"
        "  4. Country Analysis: Detailed country-specific solar patterns\n"
        "  5. Regional Comparison: Compares two specific countries\n"
        "  6. Seasonal Efficiency: Analyzes seasonal performance and variations\n"
        "- Example queries:\n"
        "  * 'Analyze daylight patterns for Germany using SARAH data'\n"
        "  * 'Compare solar potential between Spain and France using MERRA data' (use regional_comparison)\n"
        "  * 'Show geographical patterns across Europe using SARAH data' (use geographical_patterns)\n"
        "  * 'Compare Northern vs Southern Europe solar potential' (use geographical_patterns)\n"
        "  * 'What are the clear sky patterns in Italy? (Please specify SARAH or MERRA)'\n"
        "  * 'Give me a detailed country analysis for France using SARAH data'\n"
        "  * 'Analyze seasonal efficiency in Southern Europe (Please specify dataset)'\n\n"
        "  * 'What are the optimal generation hours in Spain? (Please specify dataset)'\n"
        "  * 'Compare solar potential between Northern and Southern Europe'\n\n"
        "NOTE: If the user doesn't specify whether to use SARAH or MERRA data, ask for clarification.\n"
        "SARAH is preferred for solar energy applications, while MERRA provides broader environmental context.\n\n"


        "i) Land Cover Tool (analyze_land_cover_data):\n"
        "- Analyzes historical and recent land cover data globally.\n"
        "- Functions include:\n"
        "  1. Artificial Surfaces Analysis: Retrieves coverage area of artificial surfaces (e.g., urban areas).\n"
        "  2. Shrub Cover Trends: Analyzes shrub or forest cover changes over time or across regions.\n"
        "  3. Agricultural Expansion: Identifies shifts in agricultural land use by year and country.\n"
        "  4. Regional Land Cover Comparison: Compares land cover categories between countries/regions.\n"
        "- Example queries:\n"
        "  * 'What is the area of artificial surfaces in Bolivia in 2000?'\n"
        "  * 'Show forest or shrub cover trends in Brazil between 1990 and 2020.'\n"
        "  * 'Compare land use between India and China in 2015.'\n"
        "  * 'What is the percentage of agricultural land in the United States in 2023?'\n\n"

        "j) Fuel Efficiency Tool (get_most_fuel_efficient_cars):\n"
        "- Analyzes vehicle fuel consumption and emissions data in Canada.\n"
        "- Functions include:\n"
        "  1. Most Fuel-Efficient Cars: Identifies vehicles with the lowest combined fuel consumption for a given year.\n"
        "  2. CO2 Emissions by Make: Calculates the average CO2 emissions for each car make.\n"
        "  3. Top CO2 Emitters: Lists the top 5 cars with the highest CO2 emissions.\n"
        "  4. Average Engine Size by Vehicle Class: Provides the average engine size for different vehicle classes.\n"
        "  5. Fuel Efficiency by Class: Shows the average combined fuel efficiency for different vehicle classes.\n"
        "- Example queries:\n"
        "  * 'What was the most fuel-efficient car in 2015?'\n"
        "  * 'Show me the average CO2 emissions by car make.'\n"
        "  * 'Which cars had the highest CO2 emissions in 2020?'\n"
        "  * 'What's the average engine size for SUVs?'\n"
        "  * 'Compare the average fuel efficiency of trucks and sedans.'\n\n"


        "The tool can analyze:\n"
        "- Average capacity factors by country\n"
        "- Seasonal variations (Spring=1, Summer=2, Fall=3, Winter=4)\n"
        "- Country rankings and comparisons\n"
        "- Performance relative to overall average\n\n"

        "2. PANDASAI ROUTER (query_dataset):\n"
        "Use this for ANY query that doesn't exactly match the predefined tools' capabilities. "
        "Available datasets:\n\n"
        
        "a) weather_data:\n"
        "- Contains: temperature, humidity, precipitation, wind, pressure, visibility, UV index, soil conditions\n"
        "- Use for: Complex weather analysis, correlations, patterns, comparisons\n"
        "- Example queries:\n"
        "  * 'What's the correlation between humidity and rainfall?'\n"
        "  * 'Find the least humid regions in summer'\n"
        "  * 'Compare wind speeds between different countries'\n\n"
        
        "b) fuel_data:\n"
        "- Contains: vehicle make, model, year, fuel consumption metrics\n"
        "- Use for: Complex vehicle efficiency analysis, comparisons, trends, least fuel efficient car\n"
        "- Example queries:\n"
        "  * 'Which SUV was the least efficient in 2010?'\n"
        "  * 'Compare fuel efficiency between manufacturers'\n"
        "  * 'Show fuel consumption trends for trucks'\n\n"
        
        "c) city_data, country_data, global_data:\n"
        "- Contains: Historical temperature data at different geographical levels\n"
        "- Use for: Complex temperature analysis, historical trends, geographical comparisons\n"
        "- Example queries:\n"
        "  * 'Which city had the highest average temperature?'\n"
        "  * 'Compare temperature variations between continents'\n\n"

        "d) wind_national_data:\n"
        "- Contains: National wind power capacity factors, hourly measurements, country-specific data\n"
        "- Use for: Complex wind power analysis, correlations, patterns, comparisons\n"
        "- Example queries:\n"
        "  * 'What's the correlation between German and French wind power output?'\n"
        "  * 'Show hourly wind power variations for Denmark'\n"
        "  * 'Which countries have the most consistent wind power output?'\n"
        "  * 'Compare wind power performance between neighboring countries'\n"
        "  * 'Calculate average wind power output during peak hours'\n\n"

        "e) onoffshore_wind_data:\n"
        "- Contains: Separate onshore and offshore wind power data by country\n"
        "- Use for: Complex analysis comparing onshore/offshore performance\n"
        "- Example queries:\n"
        "  * 'Compare hourly variations in onshore vs offshore output'\n"
        "  * 'Find correlation between onshore and offshore performance'\n"
        "  * 'Analyze peak performance times for different installation types'\n\n"

        "f) future_longterm_wind_data:\n"
        "- Contains: Long-term wind power projections by country\n"
        "- Use for: Complex analysis of future wind power trends\n"
        "- Example queries:\n"
        "  * 'Calculate the growth rate of wind power capacity'\n"
        "  * 'Find periods of projected peak performance'\n"
        "  * 'Project the quarter 3 power output for france next year'\n"
        "  * 'Analyze patterns by quarters in future 2 year projections for germany'\n\n"

        "g) tornado_data:\n"
        "- Contains: Detailed tornado event data including paths, damage, and characteristics\n"
        "- Use for: Complex analysis beyond predefined functions\n"
        "- Example queries for PandasAI (complex analysis):\n"
        "  * 'What's the relationship between soil moisture and tornado formation?'\n"
        "  * 'Find clusters of tornado occurrences near geographical boundaries'\n"
        "  * 'Calculate the probability of multiple tornadoes occurring on the same day'\n"
        "  * 'Analyze the correlation between tornado width and population density'\n"
        "  * 'What weather conditions preceded the most destructive tornadoes?'\n"
        "  * 'Find patterns in tornado behavior during El Niño years'\n"
        "  * 'Calculate the statistical significance of tornado path orientations'\n"
        "  * 'Identify areas with unusual tornado timing patterns'\n\n"

        "h) solar_sarah_data and solar_merra_data:\n"
        "- Contains: Solar power data from two different methodologies\n"
        "- Use for: Complex analysis beyond predefined functions\n"
        "- Example queries for PandasAI:\n"
        "  * 'Calculate the correlation between latitude and peak solar hours'\n"
        "  * 'Compare SARAH and MERRA predictions for specific regions'\n"
        "  * 'Find statistical anomalies between the two datasets'\n"
        "  * 'Calculate confidence intervals for solar predictions'\n"
        "  * 'Analyze the impact of seasonal changes on prediction accuracy'\n"
        "  * 'Identify periods where SARAH and MERRA significantly disagree'\n"
        "  * 'Calculate solar variability metrics across different timescales'\n"
        "  * 'Find optimal solar installation locations using both datasets'\n\n"

        "a) land_cover_data:\n"
        "- Contains: Land use data by category (forest, agriculture, artificial surfaces, etc.), country, year.\n"
        "- Use for: Complex land cover analysis, correlations, trends, and anomalies.\n"
        "- Example queries:\n"
        "  * 'Analyze land use change in Southeast Asia over the last decade.'\n"
        "  * 'Identify correlations between population density and urban land expansion.'\n"
        "  * 'Find regions with the highest deforestation rates.'\n\n"

        "b) fuel_data:\n"
        "- Contains: vehicle make, model, year, fuel consumption metrics.\n"
        "- Use for: Complex vehicle efficiency analysis, comparisons, trends, and correlations.\n"
        "- Example queries:\n"
        "  * 'Which SUV was the least efficient in 2010?'\n"
        "  * 'Compare fuel efficiency between manufacturers.'\n"
        "  * 'what is the model name which has the most fuel efficiency in 1999'\n\n"

        "ROUTING LOGIC:\n"
        "1. IF the query EXACTLY matches a predefined tool's capability → Use that tool\n"
        "2. IF the query requires complex analysis or doesn't match predefined functions → Use PandasAI router with appropriate dataset\n"
        "3. IF unsure → Default to PandasAI router with the most relevant dataset\n\n"

        "ROUTING LOGIC FOR WIND POWER QUERIES:\n"
        "1. Use analyze_national_wind_power when:\n"
        "   - Requesting top performing countries in wind power\n"
        "   - Analyzing seasonal patterns\n"
        "   - Comparing specific country performances\n"
        "   - Need capacity factor analysis\n\n"
        "2. Use PandasAI with wind_data when:\n"
        "   - Need complex correlations between countries\n"
        "   - Require custom time period analysis\n"
        "   - Want detailed statistical analysis\n"
        "   - Need hourly pattern analysis\n"
        "   Example queries for PandasAI:\n"
        "   * 'What's the correlation between German and French wind power output?'\n"
        "   * 'Show the hourly wind power variation for Denmark'\n"
        "   * 'Calculate the standard deviation of wind power output for each country'\n"
        "   * 'Find periods where multiple countries had peak performance'\n"
        "   * 'Analyze wind power trends during specific hours of the day'\n\n"

        "ROUTING LOGIC FOR TORNADO ANALYSIS QUERIES:\n"
        "1. Use analyze_tornado_data when:\n"
        "   - Requesting tornado frequency analysis\n"
        "   - Need detailed analysis of tornado data\n\n"

        "For EVERY response:\n"
        "1. Identify the most appropriate tool/dataset based on the query\n"
        "2. Call the function with correct parameters\n"
        "3. Provide clear results with context\n"
        "4. If using PandasAI, specify which dataset you're using and why\n\n"

        "Examples of routing:\n"
        "- 'What's the most fuel-efficient car in 2020?' → Use fuel_consumption_tool (exact match)\n"
        "- 'Which SUV had worst efficiency in 2015?' → Use PandasAI with fuel_data (complex query)\n"
        "- 'Show regions above 35°C' → Use extreme_weather_tool (exact match)\n"
        "- 'How does temperature correlate with humidity?' → Use PandasAI with weather_data (complex analysis)\n"
    )

    # Update the messages list with the new system content
    messages = [
        {
            "role": "system",
            "content": system_content
        }
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

