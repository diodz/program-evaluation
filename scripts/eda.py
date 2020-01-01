# Change made on 2024-06-26 21:06:37.985988
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests

# Pulling data from public dataset
url = "https://example.com/public_dataset.csv"
data = pd.read_csv(url)

# Preprocessing data
X = data.drop(columns=['dependent_variable_column'])
y = data['dependent_variable_column']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
