import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad


# Set random seed for reproducibility
np.random.seed(2)
tf.random.set_seed(2)

# Load dataset
df = pd.read_csv("welfarelabel.csv")

# Define outcome and treatment variable names
outcome_variable_name = 'y'
treatment_variable_name = 'w'

# Define covariate names explicitly based on the original R code
covariate_names = ['w', 'year', 'hrs1', 'hrs2', 'agewed', 'occ', 'evwork', 'wrkslf', 'wrkstat', 'wrkgovt', 'prestige', 'marital']

# Define categorical variables
categorical_vars = ['evwork', 'wrkslf', 'wrkstat', 'wrkgovt', 'marital']

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Ensure only specified covariates are used
available_covariates = [col for col in df.columns if col in covariate_names]
df = df[[outcome_variable_name] + available_covariates]

# Identify and clean problematic numeric columns
numeric_columns = ['year', 'hrs1', 'hrs2', 'agewed', 'occ', 'prestige']  # Adjust as needed
for col in numeric_columns:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.extract(r'(\d+)')  # Extract numerical part
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float

# Drop any rows with NaN values in numeric columns
df = df.dropna()

# Compute ATE
ATE = df[df['w'] == 1]['y'].mean() - df[df['w'] == 0]['y'].mean()
print(f"ATE: {ATE}")

# Split dataset into train and test sets
train_fraction = 0.80
train_df, test_df = train_test_split(df, train_size=train_fraction, random_state=2)

# Prepare feature matrices
X_train = train_df[available_covariates]
X_test = test_df[available_covariates]
y_train = train_df[outcome_variable_name]
y_test = test_df[outcome_variable_name]

# Define neural network model
model = Sequential([
    Dense(16, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=2)),
    Dense(16, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=2)),
    Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=2))
])

# Compile model
model.compile(optimizer=Adagrad(learning_rate=0.05), loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1  # Set to 1 to see progress
)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict using trained model
predictions = model.predict(X_test, batch_size=32)
ANN_error = mean_squared_error(y_test, predictions)
print(f"Neural Network MSE: {ANN_error}")

# OLS regression using Linear Regression model
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
predicted_y = ols_model.predict(X_test)
ols_error = mean_squared_error(y_test, predicted_y)
print(f"OLS MSE: {ols_error}")
