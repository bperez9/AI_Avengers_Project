# Filter data for temperatures above 40°C
extreme_temp_df = weather_data[weather_data['temperature_2m'] > 40]

# Get unique list of countries
regions = extreme_temp_df['country'].unique().tolist()

print("Regions with temperatures above 40°C:")
print(regions)

# some other stuff