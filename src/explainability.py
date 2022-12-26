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

import postprocess 
import constants


import math
import model
import json
import utils

def generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test):
    clf, feature_importances = model.get_feature_importances(X_train, Y_train)

    logger.info(f'{target} score: ' + str (clf.score(X_test,Y_test)))
    fig, ax = plt.subplots()
    feature_importances.plot.barh(ax=ax)
    ax.set_title(f"Feature importances: {target}")
    ax.set_ylabel("Mean decrease in impurity")

    plt.xticks(rotation=8)
    fig.tight_layout()
    #plt.show()

    plt.savefig(f'images/feature_importances/{target}.png')
    plt.close()
    return clf,feature_importances





TARGETS =  ["Inflation, consumer prices (annual %)",  "Pump price for diesel fuel (US$ per liter)",
"Birth rate, crude (per 1,000 people)", "Researchers in R&D (per million people)", 
    "Time required to build a warehouse (days)",
    "Wage and salaried workers, total (% of total employment) (modeled ILO estimate)","Rural population (% of total population)", 
  
    "Unemployment, total (% of total labor force) (modeled ILO estimate)", 
 "GDP per capita (constant LCU)" ,

 "Self-employed, total (% of total employment) (modeled ILO estimate)" ,
  ]

country="ESP"
dict_features_to_consider = {}
try :
    dict_features_to_consider = postprocess.get_dict_features_to_consider()
except: pass 
    

for target in TARGETS : 
    df = postprocess.process_df_target(target)
    logger.info(f"Shape: {df.shape} for:  {target} ")

    # Xtrain = df_feature_importances.drop(columns=target,axis=1) if target in df_feature_importances.columns else df_feature_importances
 
    if target in dict_features_to_consider: 
        features_to_consider = dict_features_to_consider[target]
    else: 
        features_to_consider = postprocess.calculate_features_consider( target )


    df_feature_importances = postprocess.process_df_target(target)
    serie_to_explain = postprocess.get_enriched_estimation_to_explain(country,features_to_consider)

   
    Xtrain = df_feature_importances[features_to_consider]
    Ytrain = df_feature_importances[target]
    #X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200)
    clf = model.get_model(Xtrain, Ytrain)

    df_means = pd.read_csv("data/df_means.csv")
    zone = utils.get_country_data(country)["Region"]
    df_zone = df_means.query( " `Zone` == @zone ").drop("Zone",axis=1)
    zone_features = df_zone[features_to_consider].to_dict(orient="records")[0]
    shap.initjs()
    explainer = shap.TreeExplainer(clf) 
    shap_values = explainer.shap_values(Xtrain)


    #clf_expected_value = clf.predict(pd.DataFrame([serie_to_explain.to_dict()]))    
    shap_values = explainer.shap_values(pd.DataFrame([serie_to_explain.to_dict(),zone_features]))

    shap.decision_plot(explainer.expected_value, 
            shap_values,serie_to_explain,  
            highlight=0, title=f"Predictors for estimating: {target}, {country}",
            feature_names = np.array (serie_to_explain.index),
            legend_labels=[country,zone],legend_location='lower right',
            auto_size_plot=True, link = "identity",
            feature_display_range=slice(None, -1 * constants.NUM_OF_FEATURES_TO_PLOT, -1 )) 
    a = 0   


        
# plt.title(target)
# plt.show()


# force_plot_html = shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:]).html()
# shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html}</body>"


