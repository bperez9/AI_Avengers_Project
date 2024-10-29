# tools/land_cover_tool.py

from typing import Dict
import pandas as pd
from tools.base_tool import SingleMessageTool

# Load the dataset
data_path = 'data/Land_Cover_Accounts.csv'
land_cover_data = pd.read_csv(data_path)

class GetTotalArtificialSurfaces(SingleMessageTool):
    """Tool to retrieve data about urban surfaces in a specific country."""
    
    def get_name(self) -> str:
        return "get_total_artificial_surfaces"
    
    def get_description(self) -> str:
        return "Retrieve data about urban surfaces in a specific country."
    
    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "country": {
                "description": "The name of the country, e.g. Brazil",
                "type": "string",
                "required": True
            }
        }
    
    def run_impl(self, country: str = None) -> Dict[str, str]:
        country_data = land_cover_data[land_cover_data['Country'] == country]
        if country_data.empty:
            return {"error": f"No data available for {country}."}
        
        total_artificial_surfaces = country_data[['F1992', 'F1993', 'F1994', 'F1995', 'F1996', 
                                                  'F1997', 'F1998', 'F1999', 'F2000', 'F2001',
                                                  'F2002', 'F2003', 'F2004', 'F2005', 'F2006', 
                                                  'F2007', 'F2008', 'F2009', 'F2010', 'F2011', 
                                                  'F2012', 'F2013', 'F2014', 'F2015', 'F2016', 
                                                  'F2017', 'F2018', 'F2019', 'F2020', 'F2021', 
                                                  'F2022']].sum(axis=1).values[0]
        
        return {
            "country": country,
            "total_artificial_surfaces": f"{total_artificial_surfaces:.2f} units"
        }

class GetTotalShrubCoveredAreas(SingleMessageTool):
    """Tool to retrieve data about shrub-covered areas in a specific country."""
    
    def get_name(self) -> str:
        return "get_total_shrub_covered_areas"
    
    def get_description(self) -> str:
        return "Retrieve data about shrub-covered areas in a specific country."
    
    def get_params_definition(self) -> Dict[str, dict]:
        return {
            "country": {
                "description": "The name of the country, e.g. Brazil",
                "type": "string",
                "required": True
            }
        }
    
    def run_impl(self, country: str = None) -> Dict[str, str]:
        country_data = land_cover_data[land_cover_data['Country'] == country]
        if country_data.empty:
            return {"error": f"No data available for {country}."}
        
        total_shrub_covered = country_data[['F1992', 'F1993', 'F1994', 'F1995', 'F1996', 
                                            'F1997', 'F1998', 'F1999', 'F2000', 'F2001',
                                            'F2002', 'F2003', 'F2004', 'F2005', 'F2006', 
                                            'F2007', 'F2008', 'F2009', 'F2010', 'F2011', 
                                            'F2012', 'F2013', 'F2014', 'F2015', 'F2016', 
                                            'F2017', 'F2018', 'F2019', 'F2020', 'F2021', 
                                            'F2022']].sum(axis=1).values[0]
        
        return {
            "country": country,
            "total_shrub_covered": f"{total_shrub_covered:.2f} units"
        }
