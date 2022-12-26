import pandas as pd
import os
import numpy as np 
######----------For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,5)})
plt.style.use('fivethirtyeight')
from loguru import logger

######----------For Feature Selection and Modeling
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

######---------For SHAP/Model Explainations
import shap
from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier
import xgboost as xgb
import math


NUM_OF_FEATURES_TO_PLOT=20

def get_preprocessed_df() -> pd.DataFrame :
    df = pd.read_csv("data/df_all.csv")
    if 'Unnamed: 0' in df.columns :
        del df["Unnamed: 0"]
    df =  df.dropna(axis=1, how='all')
    return df

def get_enriched_estimation_to_explain(country,feature_importances) -> pd.Series : 
    """ Get the estimation to be displayes, then fill then NaN with average values"""
    logger.info(f"calculating enriched estimation {country}" )
    average_values = get_average_values()
    df_to_explain =  get_preprocessed_df()
    serie_to_explain = df_to_explain.query( " `Country Code` == @country ").iloc[-2][feature_importances]
    for index,value in serie_to_explain.items() : 
        if math.isnan(value) :
            serie_to_explain[index] =  average_values[index] 
    return serie_to_explain

def get_average_values() ->dict :
    """ Return a dictionaty of average values"""
    df = get_preprocessed_df()
    COLUMNS_TO_REMOVE = ['Unnamed: 0',"Year", 'Country Code']
    for column in COLUMNS_TO_REMOVE: 
        if column in df.columns :
                del df[column]    
    # df = df[df[target].notna()]
    # Remove columns with all 0s
    df =  df.dropna(axis=1, how='all')
    average_values = df.mean(axis=0).to_dict()
    return average_values

def process_df_target(target):
    """ Generate df for a given target"""
    df = get_preprocessed_df()

    COLUMNS_TO_REMOVE = ['Unnamed: 0',"Year", 'Country Code']
    for column in COLUMNS_TO_REMOVE: 
        if column in df.columns :
                del df[column]

    # Remove rowd where target column is NaN
    df = df[df[target].notna()]
    # Remove columns with all 0s
    df =  df.dropna(axis=1, how='all')

    # Move target to last
    df = df.reindex(columns = [col for col in df.columns if col != target] + [target])

    # Fill with average
    for column in df.columns : 
        df[column].fillna(int(df[column].mean()), inplace=True)

    return df
