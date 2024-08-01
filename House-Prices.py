import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from Scripts import categorical_variables_2 as CAV
from Scripts import features as FT
from sklearn.pipeline import Pipeline

#Load the Data
df = pd.read_csv(r'C:\Users\adam\Documents\GitHub\House-Prices-Competition\Data\train.csv', index_col='Id')
df_test = pd.read_csv(r'C:\Users\adam\Documents\GitHub\House-Prices-Competition\Data\test.csv', index_col='Id')

#Drop the target variable
df.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = df.SalePrice
df.drop(['SalePrice'], axis=1, inplace=True)

#Split the Data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size = 0.3, random_state = 0)

#Create and Train the model
model = CatBoostRegressor(learning_rate=0.13)

clf = Pipeline(steps=[('preprocessor', CAV.Cat_var(X_train, X_valid)), ('model', model)])
clf.fit(X_train, y_train)

#Make the predictions
pred = clf.predict(X_valid)
pred_test = clf.predict(FT.XT(X_train, df_test))

print(mean_absolute_error(y_valid, pred))

#Create the submission File
output = pd.DataFrame({'Id': FT.XT(X_train, df_test).index, 'SalePrice': pred_test})
output.to_csv("File.csv", index=False)