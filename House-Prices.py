import pandas as pd
from sklearn.model_selection import train_test_split
import categorical_variables_2 as CAT
from sklearn.metrics import mean_absolute_error

# Read the data
X_full = pd.read_csv(r'C:\Users\adam\PycharmProjects\pythonProject.py\Data\train.csv', index_col='Id')
X_test_full = pd.read_csv(r'C:\Users\adam\PycharmProjects\pythonProject.py\Data\test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y,
                                                                train_size=0.8, test_size=0.2,
                                                                 random_state=0)

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

model = XGBRegressor()


my_pipeline = Pipeline(steps=[('preprocessor', CAT.Cat_var(X_train, X_valid)),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)

my_pred = my_pipeline.predict(X_valid)
print(mean_absolute_error(y_valid, my_pred))
