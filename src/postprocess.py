import pandas as pd
import os
import numpy as np 
######----------For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,5)})
plt.style.use('fivethirtyeight')
from loguru import logger
import json

######----------For Feature Selection and Modeling
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

######---------For SHAP/Model Explainations
import shap
from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier
import xgboost as xgb
import math
import model
import utils
from tqdm import tqdm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

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



TARGETS =  [ "Birth rate, crude (per 1,000 people)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)", 
 "GDP per capita (constant LCU)" ,
 "Inflation, consumer prices (annual %)" ,
 "Self-employed, total (% of total employment) (modeled ILO estimate)" ,
  ]

country="ESP"

target = TARGETS[0]



from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# compute the vif for all given features
def compute_vif(df,considered_features):
    
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

# Remove las column

def get_vif_features(df,target=None,considered_features=[]):
    """ Remove those features with a vif < 5"""
    logger.info(f"Shape: {df.shape} for:  {target} ")

    if not target :
        target = list(df.columns)[-1]

    # exclude target feature
    if target in df.columns:
        del df[target]   

    if len (considered_features) == 0 :
        considered_features = list(df.columns)

    df = df[considered_features]
    # compute vif 
    vif = compute_vif(df, considered_features).sort_values('VIF', ascending=False)

    while max(vif.VIF.values) > 5:
    # remove first column
        feature_to_remove = vif.Variable.values[0]
        logger.info(feature_to_remove)
        considered_features.remove(feature_to_remove)
        vif = compute_vif(df, considered_features).sort_values('VIF', ascending=False)

    return vif





DEFAULT_NUMBER_OF_FEATURES = 30

dict_features_to_consider  = {}
# for each of the feature
def calculate_features_consider( target , number_of_features=DEFAULT_NUMBER_OF_FEATURES, df=None):
    if not(df is not None):
        df = process_df_target(target)
    Xtrain = df.drop(columns=target,axis=1)
    Ytrain = df[target]
    clf,considered_features = model.get_feature_importances(Xtrain, Ytrain,number_of_features=number_of_features)
    vif = get_vif_features(df,target=None,considered_features=list(considered_features.index))
    return  list(vif.Variable.values)

def generate_dic_features_to_consider(number_of_features=DEFAULT_NUMBER_OF_FEATURES,df = None):

    features_to_clean =  [ "Rural population (% of total population)", 
    "Birth rate, crude (per 1,000 people)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)", 
    "GDP per capita (constant LCU)" ,
    "Inflation, consumer prices (annual %)" ,
    "Self-employed, total (% of total employment) (modeled ILO estimate)" ,
    ]

    # df_start = get_preprocessed_df()
    # features_to_clean = df_start.columns 
    for target in tqdm(features_to_clean) :
        df = process_df_target(target)
        features_to_consider = calculate_features_consider(  target , number_of_features,df)
        dict_features_to_consider[ target ] = features_to_consider

    with open('data/features_to_consider.json', 'w') as fp:
        json.dump(dict_features_to_consider, fp, sort_keys=True, indent=4)
    return dict_features_to_consider

def get_dict_features_to_consider():
    try :
        with open('data/features_to_consider.json') as json_file:
            data = json.load(json_file)
    except Exception as e :
        logger.exception(e)
        data = {}
    finally :
        return data 

def generate_zone_average_value():

    df = get_preprocessed_df()
    mappings = utils.get_country_zones_mapping()
    df['Zone'] = df['Country Code'].map(mappings) 
    zones = [ zone for zone in set(mappings.values()) if not zone != zone ] 
    results = []

    COLUMNS_TO_REMOVE = ['Unnamed: 0',"Year", 'Country Code']
    for column in COLUMNS_TO_REMOVE: 
        if column in df.columns :
                del df[column]    

    df_means  = df.groupby(['Zone']).mean()
    df_means.to_csv("data/df_means.csv")
    return df_means

#if __name__ == "__main__":
    #logger.info("generate_dic_features_to_consider")
    #generate_dic_features_to_consider() 
    #logger.info("generate_zone_average_value")
    #generate_zone_average_value()
    