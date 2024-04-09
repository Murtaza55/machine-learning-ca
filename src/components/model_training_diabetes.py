import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('notebook/data/diabetes.csv')

# X = df[['Glucose', 'BMI', 'Age']]
# y = df['Outcome']

# print(X.shape)

# To Calculate mean and standard deviation for each feature
means = df.drop('Outcome', axis=1).mean()
std_devs = df.drop('Outcome', axis=1).std()

# Define the number of standard deviations for outlier detection
num_std_devs = 2  # Adjust the number of standard deviations as needed

# Calculate lower and upper bounds for outlier detection
lower_bounds = means - ( num_std_devs * std_devs )
upper_bounds = means + ( num_std_devs * std_devs )

# Filter out input values that fall outside the lower and upper bounds
outliers = ((df.drop('Outcome', axis=1) < lower_bounds) | (df.drop('Outcome', axis=1) > upper_bounds)).any(axis=1)
df = df[~outliers]

# Split the data into features (X) and target variable (y)
X = df[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']]
y = df['Outcome']

print(X.shape)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase max_iter value
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# print(y_pred)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

pickle.dump(model, open('logistic_reg_model.pkl', 'wb'))