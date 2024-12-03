import pytest
import pandas as pd
from Tools.fuel_efficiency_tool_pandas import FuelConsumptionTool

@pytest.fixture
def sample_fuel_data():
    data = {
        'year': [2020, 2020, 2021, 2021],
        'make': ['Toyota', 'Honda', 'Ford', 'Chevrolet'],
        'model': ['Camry', 'Civic', 'F-150', 'Silverado'],
        'fuel_consumption': [7.5, 6.8, 10.5, 11.2],
        'fuel_type': ['Gasoline', 'Diesel', 'Gasoline', 'Diesel'],
        'co2_emissions': [170, 160, 250, 270]
    }
    return pd.DataFrame(data)

@pytest.fixture
def tool(sample_fuel_data):
    return FuelConsumptionTool(sample_fuel_data)

def test_run_impl_returns_correct_output(tool):
    result = tool.run_impl(2020, 'Gasoline')
    assert 'make_model' in result
    assert result['make_model'] == ['Toyota Camry']

def test_run_impl_handles_all_fuel_type(tool):
    result = tool.run_impl(2021, 'All')
    assert 'make_model' in result
    assert result['make_model'] == ['Ford F-150', 'Chevrolet Silverado']

def test_run_impl_no_results_for_invalid_year(tool):
    with pytest.raises(ValueError, match="Year must be a valid integer present in the dataset."):
        tool.run_impl(2025)

def test_run_impl_raises_error_on_missing_year_column():
    data = {
        'make': ['Toyota'],
        'model': ['Camry'],
        'fuel_consumption': [7.5],
        'fuel_type': ['Gasoline'],
        'co2_emissions': [170]
    }
    df = pd.DataFrame(data)
    tool = FuelConsumptionTool(df)
    with pytest.raises(ValueError, match="Column 'year' not found in fuel data."):
        tool.run_impl(2020)

def test_run_impl_handles_empty_dataframe():
    df = pd.DataFrame()
    tool = FuelConsumptionTool(df)
    with pytest.raises(ValueError, match="Fuel data DataFrame is empty."):
        tool.run_impl(2020)

def test_run_impl_filters_on_fuel_type(tool):
    result = tool.run_impl(2020, 'Diesel')
    assert 'make_model' in result
    assert result['make_model'] == ['Honda Civic']
