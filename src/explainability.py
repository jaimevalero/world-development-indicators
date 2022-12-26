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

NUM_OF_FEATURES_TO_PLOT=20

import math


def generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test):
    clf = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.05,
                    max_depth = 5, alpha = 10, n_estimators = 300)
    # fit the model
    clf.fit(X_train,Y_train)


    feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    

    feature_importances= feature_importances[feature_importances!=0].sort_values(ascending=False)[:NUM_OF_FEATURES_TO_PLOT].sort_values()

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


TARGETS =  [ "Birth rate, crude (per 1,000 people)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)", 
 "GDP per capita (constant LCU)" ,
 "Inflation, consumer prices (annual %)" ,
 "Self-employed, total (% of total employment) (modeled ILO estimate)" ,
  ]

country="ESP"

for target in TARGETS : 
    df = postprocess.process_df_target(target)
    logger.info(f"Shape: {df.shape} for:  {target} ")

    Xtrain = df.drop(columns=target,axis=1)
    Ytrain = df[target]
    X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200)
   
    clf,feature_importances = generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test)



    features_to_consider = list(feature_importances.to_dict().keys())

    serie_to_explain = postprocess.get_enriched_estimation_to_explain(country,features_to_consider)

    features_to_consider.append(target)
    df_feature_importances = postprocess.process_df_target(target)[features_to_consider]
    Xtrain = df_feature_importances.drop(columns=target,axis=1)
    Ytrain = df_feature_importances[target]
    X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200)
    clf,feature_importances = generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test)

    shap.initjs()
    explainer = shap.TreeExplainer(clf) 
    shap_values = explainer.shap_values(X_train)

    #clf.predict(serie_to_explain)
    clf_expected_value = clf.predict(pd.DataFrame([serie_to_explain.to_dict()]))    
    shap_values = explainer.shap_values(pd.DataFrame([serie_to_explain.to_dict()]))

    shap.decision_plot(explainer.expected_value, 
            shap_values[0,:],serie_to_explain,  highlight=0, title=target,
            feature_names = np.array (serie_to_explain.index),
            auto_size_plot=True, link = "identity",
            feature_display_range=slice(None, -1 * NUM_OF_FEATURES_TO_PLOT, -1 )) 
    a = 0   
# shap.decision_plot(explainer.expected_value, 
#         shap_values[0,:],X_train.iloc[0,:],  highlight=0, title=target,
#         feature_names = np.array (X_train.iloc[0,:].index),
#         auto_size_plot=True, link = "identity",
#         feature_display_range=slice(None, -1 * NUM_OF_FEATURES_TO_PLOT, -1 )) 
        
plt.title(target)
plt.show()


force_plot_html = shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:]).html()
shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html}</body>"


