import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.regression.linear_model import OLS
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Load the data
df_train = pd.read_csv('train_oilyML.csv')
df_train = df_train.drop(columns=["DTS"])
df_train = df_train[~(df_train == -999).any(axis=1)]
df_train = df_train.sample(frac=1).reset_index(drop=True)
# Split the training data into features and labels
X_train = df_train.drop(columns=['DTC'])
y_train = df_train['DTC']
# Create a scaler object
scaler = MinMaxScaler()

# Fit the scaler to the column
scaler.fit(df_train[['GR']])

# Transform the column
df_train['GR'] = scaler.transform(df_train[['GR']])

df_train.to_csv('submission_to_submit3.csv', index=False)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


from statsmodels.stats.stattools import durbin_watson

# Fit a linear regression model to the data
model = OLS(y_train, X_train).fit()

# Perform the Durbin-Watson test
dw_stat = durbin_watson(model.resid)

# Print the Durbin-Watson statistic
print("Durbin-Watson statistic: ", dw_stat)

# Check for autocorrelation
if dw_stat < 2 or dw_stat > 2:
    print("There is autocorrelation in the data, the model is likely non-linear")
else:
    print("There is no autocorrelation in the data, the model is likely linear")
# Define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
optimizer = Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Fit the model to the training data
model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=1)

# Make predictions on the validation set
predictions = model.predict(X_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val, predictions)

# Take the square root of the mean squared error to get the root mean squared error
rmse = np.sqrt(mse)
print("RMSE:", rmse)
# Calculate the R-squared score
r2 = r2_score(y_val, predictions)

# Print the R-squared score
print("R-squared:", r2)
df_test = pd.read_csv('test_oily.csv')
test_predictions = model.predict(df_test)
df_test['DTC'] = test_predictions
df_test = df_test[['DTC']]
df_test.to_csv('submission_oily_ML_DTC.csv', index=False)
