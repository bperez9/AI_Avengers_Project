import pytest
import pandas as pd
from Tools.sea_level_tool import SeaLevelTool  # Adjust the import path as needed

def test_calculate_average_sea_level_rise_rate():
    sea_level_data = pd.DataFrame({
        'Year': pd.to_datetime(['2000', '2001', '2002', '2003']),
        'CSIRO Adjusted Sea Level': [2.0, 2.5, 3.0, 3.5]
    })
    gsml_data = pd.DataFrame(columns=['Time', 'GMSL', 'GMSL uncertainty'])
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='rise_rate', start_year=2000, end_year=2003)
    assert result['average_rise_rate'] == '0.5000 inches per year'

def test_project_sea_level():
    sea_level_data = pd.DataFrame({
        'Year': pd.to_datetime(['2000', '2001', '2002', '2003']),
        'CSIRO Adjusted Sea Level': [2.0, 2.5, 3.0, 3.5]
    })
    gsml_data = pd.DataFrame(columns=['Time', 'GMSL', 'GMSL uncertainty'])
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='projection', end_year=2050)
    assert result['projected_sea_level'] == '27.00 inches'

def test_top_sea_level_anomalies():
    sea_level_data = pd.DataFrame({
        'Year': pd.to_datetime(['2000', '2001', '2002', '2003']),
        'CSIRO Adjusted Sea Level': [2.0, 4.0, 1.5, 3.5]
    })
    gsml_data = pd.DataFrame(columns=['Time', 'GMSL', 'GMSL uncertainty'])
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='anomalies', start_year=2000, end_year=2003, n=2)
    assert len(result['top_anomalies']) == 2
    assert result['top_anomalies'][0]['Anomaly'] == 1.25
    assert result['top_anomalies'][0]['Year'] == '2001-01-01'

def test_average_gmsl_for_year():
    sea_level_data = pd.DataFrame(columns=['Year', 'CSIRO Adjusted Sea Level'])
    gsml_data = pd.DataFrame({
        'Time': pd.to_datetime(['2000-01-01', '2000-02-01', '2000-03-01']),
        'GMSL': [1.2, 1.5, 1.3],
        'GMSL uncertainty': [0.2, 0.1, 0.3]
    })
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='average_gmsl', start_year=2000)
    assert result['average_gmsl'] == '1.33 mm'

def test_max_gmsl_uncertainty():
    sea_level_data = pd.DataFrame(columns=['Year', 'CSIRO Adjusted Sea Level'])
    gsml_data = pd.DataFrame({
        'Time': pd.to_datetime(['2000-01-01', '2001-01-01']),
        'GMSL': [1.2, 1.5],
        'GMSL uncertainty': [0.2, 0.5]
    })
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='max_uncertainty', start_year=2000, end_year=2001)
    assert result['max_uncertainty'] == '0.50 mm'

def test_empty_sea_level_data():
    sea_level_data = pd.DataFrame(columns=['Year', 'CSIRO Adjusted Sea Level'])
    gsml_data = pd.DataFrame(columns=['Time', 'GMSL', 'GMSL uncertainty'])
    tool = SeaLevelTool(sea_level_data, gsml_data)
    result = tool.run_impl(analysis_type='rise_rate', start_year=2000, end_year=2001)
    assert result['error'] == 'No data available for the specified time range'

def test_invalid_analysis_type():
    sea_level_data = pd.DataFrame({
        'Year': pd.to_datetime(['2000', '2001']),
        'CSIRO Adjusted Sea Level': [2.0, 2.5]
    })
    gsml_data = pd.DataFrame(columns=['Time', 'GMSL', 'GMSL uncertainty'])
    tool = SeaLevelTool(sea_level_data, gsml_data)
    with pytest.raises(ValueError):
        tool.run_impl(analysis_type='invalid_type')
