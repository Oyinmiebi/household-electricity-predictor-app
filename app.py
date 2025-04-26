import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = 'C:/Users/ebina/Documents/Personal/NYSC/Data_Science_Advanced/Group_2_Projects/electricity_consumption/lstm_single_household_units_model.h5'
SCALER_PATH = 'C:/Users/ebina/Documents/Personal/NYSC/Data_Science_Advanced/Group_2_Projects/electricity_consumption/scaler_single_household.pkl'
LOOKBACK_PERIOD = 72 # Must match the lookback used during training

# List of features the LSTM model was trained on (must match your training script exactly)
# This list is crucial for selecting and scaling input data correctly in the app.
LSTM_INPUT_FEATURES = ['Units', # Include the target variable itself as a lagged feature
    'global_power_rating',
    'Voltage',
    'Current',
    'num_appliances_on',
    'ac_on',
    'washing_machine_on',
    'temperature',
    'special_event',
    'holidays',
    'External_power_usage',
    'grid_fluctuation',
    # Derived time features - will be added during preprocessing
    'hour_of_day',
    'day_of_week']
N_LSTM_FEATURES = len(LSTM_INPUT_FEATURES)
TARGET_FEATURE = 'Units' # The column being predicted

# --- Helper Functions ---

# Function to load the trained model
@st.cache_resource # Cache the model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure it's in the same directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    
# Function to load the saved scaler
@st.cache_resource # Cache the scaler
def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at {scaler_path}. Please ensure it's in the same directory.")
        return None
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# Function to preprocess new data for prediction
def preprocess_data(df, scaler, lookback_period, lstm_features, target_feature):
    # Ensure data is sorted by time
    df = df.sort_values(by='time').copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Create derived time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    # Add other time features if your model used them (e.g., dayofyear, month)

    # Select the features the LSTM was trained on
    try:
        data_for_lstm = df[lstm_features]
    except KeyError as e:
        st.error(f"Missing required column(s) for LSTM input: {e}. Make sure your uploaded data has columns: {', '.join(lstm_features)}")
        return None, None, None

    # Check if there's enough data for lookback
    if len(data_for_lstm) < lookback_period:
        st.warning(f"Not enough historical data ({len(data_for_lstm)} rows) provided for a lookback of {lookback_period}. Please provide at least {lookback_period} rows.")
        return None, None, None

    # Select the last 'lookback_period' rows
    recent_data = data_for_lstm.tail(lookback_period).copy()

    # Scale the data
    try:
        scaled_data = scaler.transform(recent_data)
    except Exception as e:
        st.error(f"Error during scaling. Please check if your uploaded data format matches the scaler's expected format and features: {e}")
        return None, None, None

    # Reshape for LSTM input [n_samples, lookback, n_features]
    # We only have one sample (the last lookback_period), so n_samples = 1
    X = scaled_data.reshape(1, lookback_period, len(lstm_features))

    # Get the last timestamp for plotting reference
    last_timestamp = recent_data.index[-1]

    # Return the full processed dataframe slice for plotting history
    return X, last_timestamp, df # Return df to plot history


# Function to inverse transform predictions (for the target feature)
def inverse_transform_prediction(scaled_prediction, scaler, lstm_features, target_feature):
    # The scaler was fitted on N_LSTM_FEATURES, but we only predicted 1 (target_feature).
    # To inverse transform correctly, we need to create a dummy array with the predicted value
    # in the correct column position and other columns filled with a placeholder (like 0).
    try:
        target_col_index = lstm_features.index(target_feature)
    except ValueError:
        st.error(f"Internal error: Target feature '{target_feature}' not found in the LSTM feature list.")
        return None

    dummy_array = np.zeros((scaled_prediction.shape[0], len(lstm_features)))
    dummy_array[:, target_col_index] = scaled_prediction.flatten()

    # Inverse transform the dummy array
    original_scale_data = scaler.inverse_transform(dummy_array)

    # The target prediction is now in the correct column
    predicted_target = original_scale_data[:, target_col_index]

    return predicted_target


# --- Streamlit App Layout ---
st.title("Household Electricity Units Predictor (LSTM)")

st.write("""
Upload your historical household data (CSV format) to get predictions
for future hours using a pre-trained LSTM model.
""")

# --- Load Model and Scaler ---
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

if model is None or scaler is None:
    st.warning("Cannot proceed. Model or scaler files not found. Please ensure 'lstm_single_household_units_model.h5' and 'scaler_single_household.pkl' are in the same directory.")
    st.stop() # Stop the app if model or scaler couldn't be loaded

st.success("LSTM model and scaler loaded successfully!")


# --- User Input: Data Upload ---
uploaded_file = st.file_uploader(
    "Upload Historical Household Data (CSV)",
    type=['csv'],
    help=f"The CSV should contain at least {LOOKBACK_PERIOD} rows and have columns: time, global_power_rating, Units, Voltage, Current, num_appliances_on, ac_on, washing_machine_on, temperature, special_event, holidays, External_power_usage, grid_fluctuation."
)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("Data Preview:")
        st.dataframe(df.head())
        st.dataframe(df.tail())

        # --- User Input: Prediction Steps ---
        n_predict_steps = st.number_input(
            f"Number of future hours to predict for '{TARGET_FEATURE}'",
            min_value=1,
            max_value=168, # Predict up to one week ahead, adjust as needed
            value=24, # Default to 24 hours prediction
            step=1
        )

        # --- Prediction Button ---
        if st.button(f"Predict {TARGET_FEATURE}"):
            X, last_timestamp, df_processed = preprocess_data(
                df.copy(), scaler, LOOKBACK_PERIOD, LSTM_INPUT_FEATURES, TARGET_FEATURE
            )

            if X is not None and last_timestamp is not None and df_processed is not None:
                st.subheader(f"Prediction for the next {n_predict_steps} hours starting from {last_timestamp}")

                predicted_values = []
                current_input_sequence = X # Start with the preprocessed last lookback_period data

                current_timestamp = last_timestamp # Start timestamp for predictions

                # Multi-step prediction loop
                # This loop predicts one step, appends it, and shifts.
                # Handling future input features (temperature, appliance status, etc.)
                # is a simplification here. Ideally, you would have forecasts for these
                # or a model that predicts them as well. Here, we'll make assumptions:
                # - Time-based features (hour, day of week) are generated correctly.
                # - The model's own prediction for the target ('Units') is used in the next sequence.
                # - Other dynamic features ('temperature', 'ac_on', etc.) for future steps
                #   are assumed constant (using the last known value) or derived from a simple pattern
                #   (e.g., last known value + noise, or based on historical averages).
                #   For this demo, we'll use the last known values for simplicity for the other features.

                # Get the last known values of all input features to use as base for future steps
                last_known_features = df_processed[LSTM_INPUT_FEATURES].iloc[-1].to_dict()


                for i in range(n_predict_steps):
                    # Make prediction for the next step
                    # The model expects input shape (1, lookback, N_LSTM_FEATURES)
                    scaled_next_prediction = model.predict(current_input_sequence, verbose=0)

                    # Inverse transform the prediction (only the target feature)
                    next_prediction_target = inverse_transform_prediction(
                        scaled_next_prediction, scaler, LSTM_INPUT_FEATURES, TARGET_FEATURE
                    )
                    if next_prediction_target is None: # Handle error from inverse_transform
                         break
                    predicted_values.append(next_prediction_target[0]) # Store the single predicted value

                    # Prepare input for the next prediction step:
                    # 1. Determine the timestamp for the next step
                    current_timestamp = current_timestamp + pd.Timedelta(hours=1)

                    # 2. Generate future time-based features for this next step
                    next_hour = current_timestamp.hour
                    next_day_of_week = current_timestamp.dayofweek
                    # Add other time features here if used (dayofyear, etc.)

                    # 3. Create the data point for the next sequence
                    # This needs to include all LSTM_INPUT_FEATURES for the *next* time step.
                    next_step_features_dict = last_known_features.copy() # Start with last known values
                    next_step_features_dict['hour_of_day'] = next_hour
                    next_step_features_dict['day_of_week'] = next_day_of_week
                    next_step_features_dict[TARGET_FEATURE] = next_prediction_target[0] # Use the model's prediction for the target

                    # For other features that change (temperature, appliance status, etc.),
                    # you need a strategy for future values. Using the last known value is simple
                    # but not realistic. A better approach would involve:
                    # - Using external forecasts (e.g., weather forecast for temperature).
                    # - Modeling/predicting the other features as well.
                    # - Using historical patterns (e.g., average temperature for that time of year/day).
                    # For this demo, we are sticking to using the last known values for dynamic features
                    # other than the target and time features.

                    new_point_features = pd.DataFrame([next_step_features_dict]) # Create a tiny DataFrame
                    new_point_features = new_point_features[LSTM_INPUT_FEATURES] # Ensure column order matches training

                    # 4. Scale the new point using the same scaler
                    try:
                        scaled_new_point = scaler.transform(new_point_features)
                    except Exception as e:
                         st.error(f"Error scaling next prediction input: {e}")
                         break # Stop prediction loop on error

                    # 5. Append the scaled new point to the sequence and remove the oldest point
                    # current_input_sequence shape is (1, lookback, features)
                    # scaled_new_point shape is (1, features) -> needs to be (1, 1, features)
                    scaled_new_point_reshaped = scaled_new_point.reshape(1, 1, N_LSTM_FEATURES)

                    # Remove the first element (oldest time step) and append the new one
                    current_input_sequence = np.append(current_input_sequence[:, 1:, :], scaled_new_point_reshaped, axis=1)


                # --- Display Results ---
                if predicted_values: # Check if prediction loop completed successfully
                    st.write(f"Predicted {TARGET_FEATURE}:")
                    predicted_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=len(predicted_values), freq='h')
                    prediction_df = pd.DataFrame({f'Predicted_{TARGET_FEATURE}': predicted_values}, index=predicted_timestamps)
                    st.dataframe(prediction_df)

                    # --- Visualization ---
                    st.subheader("Historical Data and Future Prediction")
                    fig, ax = plt.subplots(figsize=(15, 6))

                    # Plot historical data (last ~2*lookback period for context)
                    historical_plot_df = df_processed[TARGET_FEATURE].tail(LOOKBACK_PERIOD * 2)
                    ax.plot(historical_plot_df.index, historical_plot_df.values, label=f'Historical {TARGET_FEATURE}')

                    # Plot predictions
                    ax.plot(prediction_df.index, prediction_df[f'Predicted_{TARGET_FEATURE}'], label=f'Predicted {TARGET_FEATURE}', linestyle='--')

                    ax.set_title(f'{TARGET_FEATURE}: Historical and Predicted')
                    ax.set_xlabel('Timestamp')
                    ax.set_ylabel(TARGET_FEATURE)
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Prediction could not be completed.")


    except Exception as e:
        st.error(f"Error processing the uploaded file or during prediction: {e}")
        import traceback
        st.error(traceback.format_exc()) # Show traceback for debugging

else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.write("Note: The multi-step prediction in this demo makes a simplifying assumption about future input features (like temperature, appliance status) by using the last known values. Real-world forecasting would require handling these features more accurately for the prediction horizon.")