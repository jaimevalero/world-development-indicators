import pandas as pd
import os
import numpy as np 


def get_country_df() :
    df = pd.read_csv("data/WDICountry.csv")
    return df

def get_country_zones_mapping():
    df = pd.read_csv("data/WDICountry.csv")
    df = df[["Country Code", "Region"]]
    df = df.set_index("Country Code")
    mappings  = df.to_dict()["Region"]
    return mappings 
    
    
def get_country_data(country):
    """ Return data for a given country"""
    df = get_country_df()
    country_dict = df.query( " `Country Code` == @country ").to_dict(orient="records")[0]
    return country_dict


def get_countries_from_same_region(country):
    """ Return countries of the same region for a given country """
    country_dict = get_country_data(country)
    region = country_dict['Region']
    all_countries = pd.read_csv("data/WDICountry.csv")
    countries = all_countries.query( " `Region`== @region ")[ "Country Code" ].values
    return countries
