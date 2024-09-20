import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

dataset_path = 'Problem3.csv'
data_df = pd.read_csv(dataset_path, sep=',')
data_df

categorical_cols = data_df.select_dtypes(include=['object','bool']).columns.to_list()

for col_name in categorical_cols:
    n_categories = data_df[col_name].nunique()
    print(f'Number of categories in {col_name}: {n_categories}')

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(data_df[categorical_cols])

encoded_categorical_df = pd.DataFrame(encoded_categorical_cols, columns=categorical_cols)

numerical_df = data_df.drop(columns=categorical_cols, axis = 1)
encoded_df = pd.concat([numerical_df, encoded_categorical_df], axis=1)
encoded_df

X = encoded_df.drop(columns= ['area'])
y = encoded_df['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

xg_reg = xgb.XGBRegressor(seed = 7,
                          learning_rate = 0.01,
                          n_estimators = 102,
                          max_depth = 3)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

data_df = pd.read_csv('Problem4.csv')
data_df

xg_class = xgb.XGBClassifier(seed=7)
xg_class.fit(X_train, y_train)

preds = xg_class.predict(X_test)

train_acc = accuracy_score(y_train, xg_class.predict(X_train))
test_acc = accuracy_score(y_test, preds)

print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')