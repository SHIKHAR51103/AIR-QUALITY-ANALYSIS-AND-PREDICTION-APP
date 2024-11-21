import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import joblib

# Paths for data and model
DATA_PATH = 'cleaned_airquality.csv'  # Replace with your dataset file name
MODEL_PATH = 'air_quality_model.joblib'  # Replace with your trained model file name

# Load dataset and model
try:
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    print("Dataset and model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure 'cleaned_airquality.csv' and 'air_quality_model.joblib' are in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset or model: {e}")
    exit()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Air Quality Monitoring Dashboard"

# App layout
app.layout = dbc.Container([
    # App Title
    dbc.Row(html.H2("Air Quality Monitoring Dashboard"), className="my-3 text-center"),

    # City Selection Dropdown
    dbc.Row([
        dbc.Col(html.Label("Select City:", className="form-label"), width=2),
        dbc.Col(
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': city, 'value': city} for city in data['City'].unique()],
                value=data['City'].unique()[0],
                placeholder="Select a city"
            ),
            width=4
        )
    ], className="mb-4"),

    # AQI Trend Chart
    dbc.Row([
        dbc.Col(dcc.Graph(id='aqi-trend-chart'), width=12)
    ]),

    # Correlation Heatmap
    dbc.Row([
        dbc.Col(html.H4("Correlation Heatmap of Pollutants and AQI"), className="mt-4"),
        dbc.Col(dcc.Graph(id='correlation-heatmap'), width=12)
    ]),

    # AQI Prediction Section
    dbc.Row(html.H4("AQI Prediction"), className="mt-4"),
    dbc.Row([
        dbc.Col(dcc.Input(id='input-pm25', type='number', placeholder="Enter PM2.5 (e.g., 35.0)", step=0.01,
                          className="form-control"), width=3),
        dbc.Col(dcc.Input(id='input-pm10', type='number', placeholder="Enter PM10 (e.g., 50.0)", step=0.01,
                          className="form-control"), width=3),
        dbc.Col(html.Button('Predict AQI', id='predict-button', n_clicks=0, className="btn btn-primary"), width=2),
        dbc.Col(html.Div(id='prediction-output', style={'marginTop': '10px', 'fontSize': '16px'}), width=4),
    ], className="mt-3"),
], fluid=True)


# Callback: AQI Trend Chart
@app.callback(
    Output('aqi-trend-chart', 'figure'),
    Input('city-dropdown', 'value')
)
def update_aqi_trend(selected_city):
    city_data = data[data['City'] == selected_city]
    fig = px.line(city_data, x='Datetime', y='AQI', title=f'AQI Trend in {selected_city}',
                  labels={'AQI': 'Air Quality Index'})
    return fig


# Callback: Correlation Heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('city-dropdown', 'value')
)
def update_heatmap(selected_city):
    try:
        # Filter data for the selected city
        city_data = data[data['City'] == selected_city]

        # Check if the city_data is empty
        if city_data.empty:
            raise ValueError(f"No data available for the selected city: {selected_city}")

        # Compute correlation matrix for numeric columns only
        numeric_columns = city_data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = city_data[numeric_columns].corr()

        # Create the heatmap figure
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title=f"Correlation Matrix for {selected_city}",
            color_continuous_scale='Viridis'
        )
        return fig
    except Exception as e:
        # Handle errors and provide feedback
        print(f"Error in heatmap callback: {e}")
        return px.imshow([[0]], text_auto=True, title="Error: Unable to generate heatmap")


@app.callback(
    Output('prediction-output', 'children'),
    [Input('input-pm25', 'value'), Input('input-pm10', 'value'), Input('predict-button', 'n_clicks')]
)
def predict_aqi(pm25, pm10, n_clicks):
    if n_clicks > 0:
        if pm25 is None or pm10 is None:
            return "Please enter values for both PM2.5 and PM10."

        try:
            # Define the complete list of features expected by the model
            feature_list = [
                'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                'Benzene', 'Toluene', 'Year', 'Month', 'Day', 'Hour',
                'PM2.5_Lag1', 'PM10_Lag1'
            ]

            # Create a dictionary with the input features
            # Default values are set to 0 for missing features, replace these with mean/median if available
            input_features = {
                'PM2.5': pm25,
                'PM10': pm10,
                'NO': 0,
                'NO2': 0,
                'NOx': 0,
                'NH3': 0,
                'CO': 0,
                'SO2': 0,
                'O3': 0,
                'Benzene': 0,
                'Toluene': 0,
                'Year': 2024,  # Use the current year or appropriate default
                'Month': 1,  # Default to January
                'Day': 1,  # Default to the 1st day
                'Hour': 0,  # Default to midnight
                'PM2.5_Lag1': 0,
                'PM10_Lag1': 0
            }

            # Ensure the input data matches the feature list
            input_data = pd.DataFrame([[input_features[feature] for feature in feature_list]], columns=feature_list)

            # Predict AQI
            prediction = model.predict(input_data)[0]
            return f"Predicted AQI: {prediction:.2f}"
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error during prediction. Please check your input values or model."

    return "Enter values and press 'Predict AQI'."


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
