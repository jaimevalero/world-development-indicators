import xgboost as xgb
import constants 
import pandas as pd
def get_model(X_train, Y_train):
    clf = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.05,
                    max_depth = 5, alpha = 10, n_estimators = 300)
    # fit the model
    clf.fit(X_train,Y_train)
    return clf


def get_feature_importances(X_train, Y_train,number_of_features=constants.NUM_OF_FEATURES_TO_PLOT):
    """ Get N feature importances for a given model"""
    clf = get_model(X_train, Y_train)

    feature_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    

    feature_importances= feature_importances[feature_importances!=0].sort_values(ascending=False)[:number_of_features].sort_values()
    return clf,feature_importances
