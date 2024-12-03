# test_extreme_weather_tool.py
import pytest
import pandas as pd
from Tools.extreme_weather_tool import ExtremeWeatherTool

#pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio
async def test_heatwave_detection():
    weather_data = pd.DataFrame({
        'temperature_2m': [36, 30, 40, 20, 37],
        'precipitation': [10, 20, 60, 5, 55],
        'country': ['CountryA', 'CountryB', 'CountryA', 'CountryC', 'CountryB']
    })
    tool = ExtremeWeatherTool(weather_data)
    result = await tool.run_impl(event_type='heatwave', threshold=35)
    assert result['regions'] == ['CountryA', 'CountryB']

@pytest.mark.asyncio
async def test_heavy_rainfall_detection():
    weather_data = pd.DataFrame({
        'temperature_2m': [36, 30, 40, 20, 37],
        'precipitation': [10, 20, 60, 5, 55],
        'country': ['CountryA', 'CountryB', 'CountryA', 'CountryC', 'CountryB']
    })
    tool = ExtremeWeatherTool(weather_data)
    result = await tool.run_impl(event_type='heavy_rainfall', threshold=50)
    assert result['regions'] == ['CountryA', 'CountryB']

@pytest.mark.asyncio
async def test_missing_temperature_column():
    weather_data = pd.DataFrame({
        'precipitation': [10, 20, 60],
        'country': ['CountryA', 'CountryB', 'CountryA']
    })
    tool = ExtremeWeatherTool(weather_data)
    with pytest.raises(ValueError, match="Column 'temperature_2m' not found in weather data."):
        await tool.run_impl(event_type='heatwave', threshold=35)

@pytest.mark.asyncio
async def test_empty_weather_data():
    weather_data = pd.DataFrame(columns=['temperature_2m', 'precipitation', 'country'])
    tool = ExtremeWeatherTool(weather_data)
    with pytest.raises(ValueError, match="Weather data DataFrame is empty."):
        await tool.run_impl(event_type='heatwave', threshold=35)

@pytest.mark.asyncio
async def test_output_format():
    weather_data = pd.DataFrame({
        'temperature_2m': [36, 30, 40, 20, 37],
        'precipitation': [10, 20, 60, 5, 55],
        'country': ['CountryA', 'CountryB', 'CountryA', 'CountryC', 'CountryB']
    })
    tool = ExtremeWeatherTool(weather_data)
    result = await tool.run_impl(event_type='heatwave', threshold=35)
    data = result['regions']
    messages = result.get('messages', [])
    assert isinstance(data, list)
    assert isinstance(messages, list)
    if data:
        assert all(isinstance(item, str) for item in data)

@pytest.mark.asyncio
async def test_invalid_threshold():
    weather_data = pd.DataFrame({
        'temperature_2m': [36, 30, 40],
        'precipitation': [10, 20, 60],
        'country': ['CountryA', 'CountryB', 'CountryA']
    })
    tool = ExtremeWeatherTool(weather_data)
    with pytest.raises(ValueError, match="Threshold must be a positive number."):
        await tool.run_impl(event_type='heatwave', threshold=-5)

@pytest.mark.asyncio
async def test_multiple_countries_weather_event():
    weather_data = pd.DataFrame({
        'temperature_2m': [36, 30, 40, 25, 37],
        'precipitation': [10, 20, 55, 60, 5],
        'country': ['CountryA', 'CountryB', 'CountryA', 'CountryC', 'CountryB']
    })
    tool = ExtremeWeatherTool(weather_data)
    result = await tool.run_impl(event_type='heatwave', threshold=35)
    regions_in_data = set(result['regions'])
    assert regions_in_data == {'CountryA', 'CountryB'}
