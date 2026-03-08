# ============================================
# 1. Import Libraries
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ============================================
# 2. Load Your Provided Dataset
# ============================================

data = pd.read_csv("hour.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# ============================================
# 3. Data Preparation
# ============================================

# Convert date column
data['dteday'] = pd.to_datetime(data['dteday'])

# Check missing values
print("\nMissing Values")
print(data.isnull().sum())

# Remove unnecessary columns
data = data.drop(columns=["instant","casual","registered"])

# ============================================
# 4. Exploratory Data Analysis
# ============================================

plt.figure(figsize=(6,4))
sns.scatterplot(x=data['temp'], y=data['cnt'])
plt.title("Temperature vs Bike Rentals")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=data['hum'], y=data['cnt'])
plt.title("Humidity vs Bike Rentals")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ============================================
# 5. Feature Engineering
# ============================================

data['year'] = data['dteday'].dt.year
data['month'] = data['dteday'].dt.month
data['dayofweek'] = data['dteday'].dt.dayofweek

# interaction feature
data['temp_humidity'] = data['temp'] * data['hum']

# ============================================
# 6. Feature Selection
# ============================================

features = [
    'season','yr','mnth','hr','holiday','weekday',
    'workingday','weathersit','temp','atemp',
    'hum','windspeed','month','dayofweek','temp_humidity'
]

X = data[features]
y = data['cnt']

# ============================================
# 7. Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 8. Train Machine Learning Model
# ============================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# 9. Model Evaluation
# ============================================

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# ============================================
# 10. Save Model
# ============================================

joblib.dump(model,"bike_rental_model.pkl")

print("Model saved successfully.")

# ============================================
# 11. Real Data Prediction Function
# ============================================

def predict_bike_rentals(
    season, yr, mnth, hr, holiday,
    weekday, workingday, weathersit,
    temp, atemp, hum, windspeed
):

    month = mnth
    dayofweek = weekday
    temp_humidity = temp * hum

    input_data = pd.DataFrame([[

        season, yr, mnth, hr, holiday,
        weekday, workingday, weathersit,
        temp, atemp, hum, windspeed,
        month, dayofweek, temp_humidity

    ]], columns=features)

    prediction = model.predict(input_data)

    return int(prediction[0])



result = predict_bike_rentals(

    season=2,        # summer
    yr=1,            # 2012
    mnth=6,          # June
    hr=14,           # 2 PM
    holiday=0,
    weekday=3,
    workingday=1,
    weathersit=1,    # clear
    temp=0.72,
    atemp=0.70,
    hum=0.60,
    windspeed=0.20

)

print("\nPredicted Bike Rentals:", result)