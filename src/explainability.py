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



def get_preprocessed_df() -> pd.DataFrame :
    df = pd.read_csv("data/df_all.csv")
    if 'Unnamed: 0' in df.columns :
        del df["Unnamed: 0"]
    df =  df.dropna(axis=1, how='all')
    return df




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

def generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test):
    clf = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.05,
                    max_depth = 5, alpha = 10, n_estimators = 300)
    # fit the model
    clf.fit(X_train,Y_train)


    feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    NUM_OF_FEATURES_TO_PLOT=12

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
    return clf,feature_importances


TARGETS =  [ "GDP per capita (constant LCU)" ,
 "Inflation, consumer prices (annual %)" ,
 "Self-employed, total (% of total employment) (modeled ILO estimate)" ,
 "Unemployment, total (% of total labor force) (modeled ILO estimate)" ]


for target in TARGETS : 
    df = process_df_target(target)
    logger.info(f"Shape: {df.shape} for:  {target} ")

    Xtrain = df.drop(columns=target,axis=1)
    Ytrain = df[target]
    X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200)
   
    clf,feature_importances = generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test)

    #shap.initjs()
    #explainer = shap.TreeExplainer(clf) 
    #shap_values = explainer.shap_values(X_train)
    #shap.force_plot(explainer.expected_value[0])   
    #shap.summary_plot(shap_values, X_test) 
    
    # clf0 =  RandomForestRegressor(max_depth=2, random_state=0)
    # clf0.fit(X_train,Y_train)
    # shap.initjs()
    # explainer = shap.TreeExplainer(clf0)
    # a = 0



    features_to_consider = list(feature_importances.to_dict().keys())
    features_to_consider.append(target)
    df_feature_importances = process_df_target(target)[features_to_consider]
    Xtrain = df_feature_importances.drop(columns=target,axis=1)
    Ytrain = df_feature_importances[target]
    X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200)
    clf,feature_importances = generate_feature_importance_images(target, X_train, X_test, Y_train, Y_test)

    shap.initjs()
    explainer = shap.TreeExplainer(clf) 
    shap_values = explainer.shap_values(X_train)
    #shap.force_plot(explainer.expected_value, shap_values[0,:],  X_train.iloc[0,:])
    shap.force_plot(explainer.expected_value, shap_values[0,:],  
        X_train.iloc[0,:],
        matplotlib=True, 
        show=False
        title=target)
    plt.savefig('force_plot.png')

    a =0 
