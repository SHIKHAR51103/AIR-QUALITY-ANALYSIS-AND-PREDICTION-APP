# Air Quality Monitoring and Analysis Script
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Data Preprocessing
def preprocess_data(file_path):
    data = pd.read_csv("airquality.csv")
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    columns_to_impute = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
    data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])
    data.drop(columns=['Xylene'], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['City'] = label_encoder.fit_transform(data['City'])

    aqi_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    data['AQI_Bucket'] = data['AQI_Bucket'].astype('category').cat.set_categories(aqi_order, ordered=True)
    data['AQI_Bucket'] = data['AQI_Bucket'].cat.codes

    # Feature engineering
    data['Year'] = data['Datetime'].dt.year
    data['Month'] = data['Datetime'].dt.month
    data['Day'] = data['Datetime'].dt.day
    data['Hour'] = data['Datetime'].dt.hour
    data['PM2.5_Lag1'] = data['PM2.5'].shift(1)
    data['PM10_Lag1'] = data['PM10'].shift(1)
    data.dropna(inplace=True)

    output_file = 'cleaned_airquality.csv'
    data.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Cleaned data saved to '{output_file}'.")
    return data

# Data Visualization
def visualize_data(data):
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
    plt.figure(figsize=(15, 10))
    for i, pollutant in enumerate(pollutants, 1):
        plt.subplot(4, 3, i)
        sns.histplot(data[pollutant], kde=True, bins=30)
        plt.title(f'Distribution of {pollutant}')
    plt.tight_layout()
    plt.show()

    corr_matrix = data[pollutants + ['AQI']].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Pollutants and AQI')
    plt.show()

# Model Training and Evaluation
def train_model(data):
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene',
                'Year', 'Month', 'Day', 'Hour', 'PM2.5_Lag1', 'PM10_Lag1']
    target = 'AQI'

    sample_data = data.sample(frac=0.1, random_state=42)
    X = sample_data[features]
    y = sample_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Model Performance:\nMean Absolute Error: {mae}\nRoot Mean Squared Error: {rmse}")

    model_file = 'air_quality_model.joblib'
    joblib.dump(model, model_file)
    print(f"Model saved as '{model_file}'.")
    return model

# Main Execution
if __name__ == '__main__':
    dataset_path = 'airquality.csv'  # Update the path to your dataset file
    if not os.path.exists(dataset_path):
        print(f"Dataset file '{dataset_path}' not found. Please place the file in the script directory.")
    else:
        cleaned_data = preprocess_data(dataset_path)
        visualize_data(cleaned_data)
        trained_model = train_model(cleaned_data)
