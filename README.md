
#  Bike Rental Demand Prediction

A machine learning project that predicts bike rental demand using weather and time-based features. The model analyzes historical bike-sharing data to learn patterns between environmental conditions and rental activity.

This project demonstrates a complete **data science workflow**, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

---

# Project Objective

The objective of this project is to build a machine learning model capable of predicting the number of bike rentals based on:

* Weather conditions
* Seasonal patterns
* Time-related features

Accurate demand prediction can help bike-sharing systems improve **resource allocation, station balancing, and operational efficiency**.

---

# Dataset

The project uses the **Bike Sharing Dataset**, which contains historical bike rental records along with weather and temporal information.

Main dataset files:

* `hour.csv` – hourly bike rental data
* `day.csv` – daily bike rental data

Important features include:

| Feature    | Description                  |
| ---------- | ---------------------------- |
| season     | Season of the year           |
| yr         | Year (0 = 2011, 1 = 2012)    |
| mnth       | Month                        |
| hr         | Hour of the day              |
| holiday    | Whether the day is a holiday |
| weekday    | Day of the week              |
| workingday | Working day indicator        |
| weathersit | Weather condition            |
| temp       | Normalized temperature       |
| atemp      | Feels-like temperature       |
| hum        | Humidity                     |
| windspeed  | Wind speed                   |
| cnt        | Total bike rentals           |

---

# ⚙️ Project Workflow

The project follows a structured machine learning pipeline.

## 1️⃣ Data Preparation

* Load dataset
* Convert date fields
* Check missing values
* Remove unnecessary columns

## 2️⃣ Exploratory Data Analysis (EDA)

* Analyze relationships between weather variables and bike rentals
* Create scatter plots and correlation heatmaps
* Identify patterns and trends

## 3️⃣ Feature Engineering

New features were created to improve prediction performance:

* Extract **year, month, and day of week**
* Create interaction feature:

```
temp_humidity = temp × hum
```

These features help capture hidden relationships in the data.

## 4️⃣ Model Development

The dataset was split into training and testing sets.

```
80% Training Data
20% Testing Data
```

The machine learning model used:

**Random Forest Regressor**

Random Forest was chosen because it performs well with nonlinear relationships and complex datasets.

## 5️⃣ Model Evaluation

The model was evaluated using:

* **Mean Squared Error (MSE)**
* **R² Score**

These metrics measure prediction accuracy and how well the model explains variance in the data.

---

# Example Prediction

The trained model can predict bike rentals using real input data.

Example input:

```
Season: Summer
Month: June
Hour: 2 PM
Weather: Clear
Temperature: 0.72
Humidity: 0.60
Wind Speed: 0.20
```

Example output:

```
Predicted Bike Rentals: 325
```

This indicates approximately **325 bikes may be rented during that hour**.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Project Structure

```
Bike-Rental-Prediction
│
├── hour.csv
├── day.csv
├── bike_rental_prediction_notebook.ipynb
├── README.md
└── requirements.txt
```

---

# How to Run the Project

1️⃣ Clone the repository

```
git clone https://github.com/yourusername/bike-rental-prediction.git
```

2️⃣ Navigate to the project folder

```
cd bike-rental-prediction
```

3️⃣ Install dependencies

```
pip install -r requirements.txt
```

4️⃣ Run the Jupyter Notebook

```
jupyter notebook
```

Open the notebook and run all cells to train the model and generate predictions.

---

# 📈 Future Improvements

Possible improvements include:

* Integrating **real-time weather API**
* Using **time series forecasting models**
* Building a **web application for live predictions**
* Hyperparameter tuning for better performance

---

#  License

This project is open-source and available under the MIT License.

