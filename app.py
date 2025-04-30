import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# App title and description
st.title("⚡ Household Electricity Consumption Predictor")
st.markdown("""
This app predicts future electricity consumption based on historical data using an LSTM neural network model.
Upload your recent data to see predictions for your specified time horizon.
""")

@st.cache_resource
def load_model_and_scaler():
    """Load the saved model and scaler, and return them."""
    try:
        # Define custom objects dictionary, mapping 'mse' to the MeanSquaredError loss function
        # Also include MeanSquaredError itself, just in case it was saved by class name
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError,
            'MeanSquaredError': tf.keras.losses.MeanSquaredError
        }

        # Load the model with custom objects
        # Make sure these paths are correct on your system
        model_path = "lstm_single_household_units_model.h5"
        scaler_path = 'scaler_single_household.pkl'

        if not os.path.exists(model_path):
             st.error(f"Model file not found at {model_path}. Please ensure the file exists.")
             return None, None, False

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )

        if not os.path.exists(scaler_path):
             st.error(f"Scaler file not found at {scaler_path}. Please ensure the file exists.")
             return None, None, False

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        return model, scaler, True

    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, False

# Load model and scaler
model, scaler, model_loaded = load_model_and_scaler()

# Define the features used by the model
LSTM_INPUT_FEATURES = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'hour_of_day',
    'day_of_week'
]

TARGET_FEATURE = 'Global_active_power'
LOOKBACK_PERIOD = 72  # Number of past hours used for prediction

# Function to prepare data for prediction
def prepare_data_for_prediction(df):
    """Prepare data for prediction by the LSTM model."""
    # Add time features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Check if required features exist in the dataframe
    # Create a list of required features excluding the derived time features for the check
    required_features_check = [f for f in LSTM_INPUT_FEATURES if f not in ['hour_of_day', 'day_of_week']]

    if not all(feature in df.columns for feature in required_features_check):
        missing_features = [feature for feature in required_features_check if feature not in df.columns]
        st.error(f"Uploaded data is missing required columns: {', '.join(missing_features)}. Please upload a CSV with all necessary columns.")
        return None

    # Select and reorder columns to match the model's expected input order
    # Ensure all LSTM_INPUT_FEATURES are present before selecting
    if not all(feature in df.columns for feature in LSTM_INPUT_FEATURES):
         # This case should ideally be caught by the previous check, but as a safeguard
         missing_features_select = [f for f in LSTM_INPUT_FEATURES if f not in df.columns]
         st.error(f"Internal error: Cannot select all LSTM input features after creating time features. Missing: {', '.join(missing_features_select)}")
         return None

    df_ordered = df[LSTM_INPUT_FEATURES]

    # Handle potential missing values *before* scaling
    if df_ordered.isnull().sum().sum() > 0:
        st.warning("Missing values found in uploaded data. Attempting interpolation and filling.")
        try:
            # Use limit_direction='both' for interpolation to fill NaNs at the beginning/end
            df_ordered = df_ordered.interpolate(method='time', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
            if df_ordered.isnull().sum().sum() > 0:
                 st.error("Missing values still remain after filling. Cannot proceed with prediction.")
                 return None
        except Exception as e:
             st.error(f"Error handling missing values: {e}")
             return None


    # Check if there is enough data for the lookback period
    if len(df_ordered) < LOOKBACK_PERIOD:
        st.error(f"Insufficient data for prediction. Need at least {LOOKBACK_PERIOD} hours of historical data, but received only {len(df_ordered)}.")
        return None


    # Scale the data
    try:
        scaled_data = scaler.transform(df_ordered)
        scaled_df = pd.DataFrame(scaled_data, index=df_ordered.index, columns=LSTM_INPUT_FEATURES)
        return scaled_df
    except Exception as e:
        st.error(f"Error scaling data: {e}. Ensure uploaded data format matches the scaler's expected features and order used during training.")
        return None


# Function to make predictions
def predict_future(df_scaled, hours_ahead=24):
    """Make predictions for the specified number of hours ahead."""
    predictions = []
    # Use the last LOOKBACK_PERIOD from the scaled historical data to start
    current_sequence = df_scaled.iloc[-LOOKBACK_PERIOD:].values.copy() # Use .copy() to avoid SettingWithCopyWarning later

    # Get the last historical timestamp to start generating future timestamps
    last_historical_time = df_scaled.index[-1]

    # Get the indices of time features and target feature for filling the next step
    try:
        hour_index = LSTM_INPUT_FEATURES.index('hour_of_day')
        dayofweek_index = LSTM_INPUT_FEATURES.index('day_of_week')
        target_index = LSTM_INPUT_FEATURES.index(TARGET_FEATURE)
        # Identify features to carry forward from the last historical point
        features_to_carry_forward_indices = [
            LSTM_INPUT_FEATURES.index(f) for f in LSTM_INPUT_FEATURES
            if f not in [TARGET_FEATURE, 'hour_of_day', 'day_of_week']
        ]

    except ValueError as e:
        st.error(f"Internal error: Required feature index not found: {e}")
        return None


    for i in range(hours_ahead):
        # Reshape for LSTM (batch_size, time_steps, features)
        current_sequence_reshaped = current_sequence.reshape(1, LOOKBACK_PERIOD, len(LSTM_INPUT_FEATURES))

        # Predict next value (the target feature)
        try:
            next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predictions.append(next_pred_scaled)
        except Exception as e:
            st.error(f"Error during model prediction at step {i+1}: {e}")
            return None # Stop prediction if an error occurs


        # Prepare input for the next prediction step (multi-step forecasting)
        next_timestamp = last_historical_time + pd.Timedelta(hours=i+1)

        # Create a new scaled feature vector for the next timestamp
        new_point_scaled = np.zeros(len(LSTM_INPUT_FEATURES))

        # Set the predicted target feature value (scaled)
        new_point_scaled[target_index] = next_pred_scaled

        # Set the time features for the next step. Assuming time features were NOT scaled.
        new_point_scaled[hour_index] = next_timestamp.hour
        new_point_scaled[dayofweek_index] = next_timestamp.weekday()

        # For other features, carry forward the *last known scaled values* from the previous sequence step.
        # This is a simplification; ideally, these would also be predicted or come from external forecasts.
        # The last row of the *current_sequence* contains the most recent features used for the prediction.
        last_features_in_sequence_scaled = current_sequence[-1, :]
        for idx in features_to_carry_forward_indices:
             new_point_scaled[idx] = last_features_in_sequence_scaled[idx]


        # Update the sequence for next prediction (remove oldest, add newest)
        current_sequence = np.vstack([current_sequence[1:], new_point_scaled])


    return predictions


# Function to convert predictions back to original scale
def rescale_predictions(predictions_scaled, scaler, target_feature_index, n_features):
    """Convert predictions from scaled values back to original scale."""
    if not predictions_scaled: # Handle case where predictions list is empty
        return np.array([])

    # Initialize an array of zeros with the shape expected by the scaler
    # The shape should be (number of samples, number of features the scaler was fitted on)
    dummy_array = np.zeros((len(predictions_scaled), n_features))

    # Place the scaled predictions into the target feature's column
    # Ensure predictions_scaled is a numpy array for correct slicing
    dummy_array[:, target_feature_index] = np.array(predictions_scaled).flatten()


    # Inverse transform the dummy array
    try:
        original_scale_data = scaler.inverse_transform(dummy_array)
        # Extract the target feature's column from the inverse-transformed data
        original_scale_preds = original_scale_data[:, target_feature_index]
        return original_scale_preds
    except Exception as e:
        st.error(f"Error during inverse transformation: {e}")
        return None


# Function to create future timestamps
def create_future_timestamps(last_timestamp, hours_ahead):
    """Create timestamps for future predictions."""
    # Ensure last_timestamp is a datetime object
    if not isinstance(last_timestamp, (pd.Timestamp, datetime.datetime)):
         st.error("Last timestamp is not a valid datetime object.")
         return []

    future_timestamps = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(hours_ahead)]
    return future_timestamps


# Function to create interactive visualizations
# REMOVED: create_prediction_chart function as requested
# def create_prediction_chart(historical_data, predicted_values, future_timestamps):
#     """Create an interactive chart with historical data and predictions."""
#     # ... (function definition removed) ...
#     pass # Or return go.Figure() as a placeholder if needed elsewhere

# Function to create daily pattern analysis
def create_daily_pattern_chart(data):
    """Create a chart showing daily consumption patterns."""
    if data is None or data.empty or TARGET_FEATURE not in data.columns:
        return go.Figure() # Return empty figure if data is insufficient

    # Ensure index is datetime before grouping by hour
    if not isinstance(data.index, pd.DatetimeIndex):
         st.warning("Data index is not datetime, cannot create daily pattern chart.")
         return go.Figure()

    hourly_avg = data.groupby(data.index.hour)[TARGET_FEATURE].mean()

    fig = px.line(
        x=hourly_avg.index,
        y=hourly_avg.values,
        labels={'x': 'Hour of Day', 'y': f"Average {TARGET_FEATURE.replace('_', ' ')}"},
        title="Average Consumption by Hour of Day"
    )

    fig.update_layout(height=400)
    return fig

# Function to create weekly pattern analysis
def create_weekly_pattern_chart(data):
    """Create a chart showing weekly consumption patterns."""
    if data is None or data.empty or TARGET_FEATURE not in data.columns:
         return go.Figure() # Return empty figure if data is insufficient

    # Ensure index is datetime before grouping by day of week
    if not isinstance(data.index, pd.DatetimeIndex):
         st.warning("Data index is not datetime, cannot create weekly pattern chart.")
         return go.Figure()

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = data.groupby(data.index.dayofweek)[TARGET_FEATURE].mean()

    fig = px.bar(
        x=[day_names[i] for i in daily_avg.index],
        y=daily_avg.values,
        labels={'x': 'Day of Week', 'y': f"Average {TARGET_FEATURE.replace('_', ' ')}"},
        title="Average Consumption by Day of Week"
    )

    fig.update_layout(height=400)
    return fig

# Function to create monthly pattern analysis
def create_monthly_pattern_chart(data):
    """Create a chart showing monthly consumption patterns."""
    if data is None or data.empty or TARGET_FEATURE not in data.columns:
         return go.Figure() # Return empty figure if data is insufficient

    # Ensure index is datetime before grouping by month
    if not isinstance(data.index, pd.DatetimeIndex):
         st.warning("Data index is not datetime, cannot create monthly pattern chart.")
         return go.Figure()

    # Group by month and calculate the mean
    monthly_avg = data.groupby(data.index.to_period('M'))[TARGET_FEATURE].mean()
    # Convert Period index to string for Plotly
    monthly_avg.index = monthly_avg.index.astype(str)


    fig = px.line(
        x=monthly_avg.index,
        y=monthly_avg.values,
        labels={'x': 'Month', 'y': f"Average {TARGET_FEATURE.replace('_', ' ')}"},
        title="Average Consumption by Month"
    )

    fig.update_layout(height=400)
    return fig


# Function to provide Decision Support recommendations (Enhanced)
def get_dss_recommendations(predicted_power_kwh_array, future_timestamps, current_balance_naira, cost_per_unit_naira):
    """Provides enhanced decision support recommendations based on predicted usage pattern and balance."""
    recommendations = []

    if cost_per_unit_naira <= 0:
        recommendations.append("Please provide a valid cost per unit to get budget and usage recommendations.")
        return recommendations

    total_predicted_power_kwh = np.sum(predicted_power_kwh_array)
    forecast_hours = len(predicted_power_kwh_array)
    estimated_cost_naira = total_predicted_power_kwh * cost_per_unit_naira
    units_in_balance = current_balance_naira / cost_per_unit_naira

    recommendations.append(f"Predicted total energy consumption over the next {forecast_hours} hours: **{total_predicted_power_kwh:,.2f} kWh**")
    recommendations.append(f"Estimated cost for this predicted usage: **₦{estimated_cost_naira:,.2f}**")
    recommendations.append(f"Your current balance (₦{current_balance_naira:,.2f}) is equivalent to approximately **{units_in_balance:,.2f} kWh**.")

    # --- Budget Recommendations ---
    recommendations.append("--- Budget Recommendations ---")
    if total_predicted_power_kwh > units_in_balance:
        units_needed_strictly = total_predicted_power_kwh - units_in_balance
        recommended_units_to_buy = units_needed_strictly * 1.15 # 15% buffer

        recommendations.append(f"⚠️ Your predicted usage is likely to **exceed** your current balance equivalent units.")
        recommendations.append(f"**Recommendation:** Consider buying approximately **{recommended_units_to_buy:,.2f} kWh** (₦{recommended_units_to_buy * cost_per_unit_naira:,.2f}) to cover the predicted usage plus a buffer.")
    elif total_predicted_power_kwh <= units_in_balance and total_predicted_power_kwh > units_in_balance * 0.8:
         recommendations.append(f"🗸 Your current balance appears sufficient, but you are close to your limit.")
         recommendations.append("Consider topping up soon if you anticipate higher future usage or for peace of mind.")
    else:
         recommendations.append(f"✅ Your current balance is well above the predicted consumption for this period.")
         recommendations.append(f"You're doing good on budget!")

    # --- Usage Pattern Recommendations ---
    recommendations.append("--- Usage Pattern Insights ---")

    if len(predicted_power_kwh_array) > 1:
        # Find peak consumption time(s)
        max_power_kwh = np.max(predicted_power_kwh_array)
        peak_hours_indices = np.where(predicted_power_kwh_array == max_power_kwh)[0]

        if peak_hours_indices.size > 0:
             peak_timestamps = [future_timestamps[i] for i in peak_hours_indices]
             peak_times_str = ", ".join([ts.strftime('%Y-%m-%d %H:%M') for ts in peak_timestamps])
             recommendations.append(f"⚡️ Peak predicted consumption ({max_power_kwh:.2f} kW) is expected around: **{peak_times_str}**")
             recommendations.append("Consider reducing simultaneous usage of heavy appliances during these times to potentially lower your overall consumption and avoid overloading circuits.")
        else:
             recommendations.append("No clear peak predicted consumption identified in this period.")

        # Basic analysis of variability
        std_dev_power = np.std(predicted_power_kwh_array)
        recommendations.append(f"The predicted usage varies (Standard Deviation: {std_dev_power:.2f} kW). Understanding when you use the most power can help optimize consumption.")


    else:
        recommendations.append("Forecast period is too short to provide detailed usage pattern insights.")


    return recommendations


# Main app layout
if model_loaded:
    st.sidebar.header("Prediction Settings")

    # Add DSS Input fields in the sidebar
    st.sidebar.subheader("Decision Support Inputs")
    cost_per_unit_naira = st.sidebar.number_input(
        "Estimated Cost Per kWh (₦)",
        min_value=0.0,
        value=50.0, # Arbitrary default value in Naira per kWh
        step=0.1,
        format="%.2f",
        help="Enter the estimated cost in Naira for 1 kWh of electricity."
    )

    current_balance_naira = st.sidebar.number_input(
        "Current Meter Balance (₦)",
        min_value=0.0,
        value=5000.0, # Arbitrary default balance in Naira
        step=100.0,
        format="%.2f",
        help="Enter your current balance on your prepaid meter in Naira."
    )
    st.sidebar.markdown("---")


    # Always expect uploaded data if demo option is removed
    uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV or TXT)", type=["csv", "txt"])

    data = None # Initialize data to None

    if uploaded_file is not None:
        try:
            # Try to parse the uploaded file
            # Assuming the structure is similar to the Individual Household Electric Power Consumption dataset
            data = pd.read_csv(uploaded_file, sep=';',
                             parse_dates={'dt': ['Date', 'Time']}, # Corrected back to 'Time' based on user's likely data
                             infer_datetime_format=True,
                             low_memory=False,
                             na_values=['nan', '?'],
                             index_col='dt')

            # Convert columns to numeric, coercing errors
            for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
                 if col in data.columns:
                     data[col] = pd.to_numeric(data[col], errors='coerce')


            # Handle missing values - after converting to numeric
            if data is not None and not data.empty and data.isnull().sum().sum() > 0:
                st.warning("Missing values found in uploaded data. Attempting interpolation and filling.")
                try:
                    # Use limit_direction='both' for interpolation to fill NaNs at the beginning/end
                    data = data.interpolate(method='time', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
                    if data.isnull().sum().sum() > 0:
                         st.error("Missing values still remain after filling. Cannot proceed with prediction.")
                         data = None # Set data to None if NaNs persist
                except Exception as e:
                     st.error(f"Error handling missing values: {e}")
                     data = None # Set data to None on error

            if data is not None and data.empty:
                 st.error("Uploaded data is empty after processing.")
                 data = None # Set data to None if empty


        except Exception as e:
            st.sidebar.error(f"Error reading or processing file: {e}")
            import traceback
            st.sidebar.error(traceback.format_exc()) # Show traceback for debugging
            data = None

    else:
         st.info("Please upload a historical data file to get started.")


    # User inputs for prediction
    hours_ahead = st.sidebar.slider("Hours to Predict Ahead", min_value=1, max_value=168, value=24,
                                   help="Number of hours you want to predict into the future")

    # Add a predict button
    predict_button = st.sidebar.button("Generate Forecast")

    # Use tabs for different sections of the output
    forecast_tab, advanced_analytics_tab = st.tabs(["Forecast Results", "Consumption Patterns"])


    # Display data and predictions if data is available
    if data is not None:

        # Content for the Forecast tab
        with forecast_tab:
            if predict_button:
                 # Check if there is enough data for the lookback period BEFORE prediction
                 if len(data) < LOOKBACK_PERIOD:
                      st.error(f"Insufficient data for prediction. Need at least {LOOKBACK_PERIOD} hours of historical data, but received only {len(data)}.")
                 else:
                      st.subheader("Electricity Consumption Forecast")

                      with st.spinner("Generating forecast..."):
                          # Prepare data
                          data_prepared = prepare_data_for_prediction(data.copy())


                          if data_prepared is not None:
                              target_feature_index_in_lstm_features = LSTM_INPUT_FEATURES.index(TARGET_FEATURE)
                              scaled_predictions = predict_future(data_prepared, hours_ahead)

                              if scaled_predictions is not None:
                                  predictions_original = rescale_predictions(scaled_predictions, scaler, target_feature_index_in_lstm_features, len(LSTM_INPUT_FEATURES))

                                  if predictions_original is not None:
                                      future_times = create_future_timestamps(data.index[-1], hours_ahead)

                                      # Create a dataframe with predictions (used for download)
                                      predictions_df = pd.DataFrame({
                                          'Timestamp': future_times,
                                          'Predicted Power': predictions_original
                                      }).set_index('Timestamp')

                                      # Use a single column now that the chart is removed
                                      with st.container(): # Using a container to group summary/download/DSS
                                          st.write("Prediction Summary")
                                          if isinstance(predictions_original, np.ndarray) and predictions_original.size > 0:
                                              # Show Max Predicted Power in summary
                                              st.metric("Max Predicted Power", f"{predictions_original.max():.2f} kW")

                                              # Calculate Total Predicted Power for DSS (still needed)
                                              total_predicted_power_kwh = predictions_original.sum()
                                              # Optionally display total power if user wants to see it separate from summary
                                              st.metric(f"Total Predicted {TARGET_FEATURE.replace('_', ' ')} ({hours_ahead} hours)", f"{total_predicted_power_kwh:.2f} kWh (Total)") # Display total in kWh and clarify


                                          else:
                                               st.info("No predictions generated or summary available.")

                                      # Download button for predictions (remains visible)
                                      # Moved outside the col2 block to span full width if col1 is removed or adjusted
                                      if predictions_df is not None and not predictions_df.empty:
                                           csv = predictions_df.to_csv()
                                           st.download_button(
                                               label="Download Detailed Forecast CSV",
                                               data=csv,
                                               file_name="electricity_forecast_detailed.csv",
                                               mime="text/csv",
                                           )
                                      else:
                                           st.info("No detailed forecast data to download.")


                                      # --- Decision Support Section (Enhanced) ---
                                      st.subheader("Decision Support Recommendations")
                                      if isinstance(predictions_original, np.ndarray) and predictions_original.size > 0 and future_times:
                                           dss_recommendations = get_dss_recommendations(
                                               predictions_original, # Pass the full array
                                               future_times, # Pass future timestamps for context
                                               current_balance_naira,
                                               cost_per_unit_naira
                                           )
                                           for rec in dss_recommendations:
                                               if "⚠️" in rec:
                                                   st.warning(rec)
                                               elif "🗸" in rec or "✅" in rec:
                                                   st.success(rec)
                                               elif "⚡️" in rec: # Highlight peak time recommendation
                                                   st.info(rec)
                                               elif "---" in rec: # Use markdown for separators
                                                    st.markdown(rec)
                                               else:
                                                   st.write(rec)
                                      else:
                                           st.info("Decision support recommendations require a generated forecast and valid inputs.")
                                      # --- END Decision Support ---


                                  else:
                                      st.error("Failed to rescale predictions.")
                              else:
                                  st.error("Failed to generate predictions.")
                          else:
                              st.error("Data preparation for prediction failed.")
            else:
                 # Display basic data info when predict button is not clicked in Forecast tab
                 st.subheader("Sample of Historical Data")
                 st.dataframe(data.tail(10))

                 st.subheader("Data Statistics")
                 col1, col2 = st.columns(2) # Keep 2 columns for data stats layout
                 with col1:
                     st.metric("Number of Records", f"{len(data):,}")
                     if isinstance(data.index, pd.DatetimeIndex):
                          st.metric("Date Range", f"{data.index.min().date()} to {data.index.max().date()}")
                     else:
                          st.info("Cannot display date range (index is not datetime).")
                 with col2:
                     if TARGET_FEATURE in data.columns and not data.empty:
                          st.metric("Average Power", f"{data[TARGET_FEATURE].mean():.2f} kW")
                          st.metric("Max Power", f"{data[TARGET_FEATURE].max():.2f} kW")
                     else:
                          st.info("Cannot display power statistics (missing target column or data).")


        # Content for the Advanced Analytics tab
        with advanced_analytics_tab:
             st.subheader("Consumption Patterns")
             col1, col2 = st.columns(2) # Keep 2 columns for chart layout

             with col1:
                 daily_fig = create_daily_pattern_chart(data)
                 if daily_fig:
                      st.plotly_chart(daily_fig, use_container_width=True)

             with col2:
                 weekly_fig = create_weekly_pattern_chart(data)
                 if weekly_fig:
                      st.plotly_chart(weekly_fig, use_container_width=True)

             # Monthly Pattern Chart
             st.markdown("---") # Separator
             monthly_fig = create_monthly_pattern_chart(data)
             if monthly_fig:
                  st.plotly_chart(monthly_fig, use_container_width=True)


    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Model")
    st.sidebar.info(
        "This application uses an LSTM neural network model to predict household electricity consumption. "
        "The model was trained on historical data and considers various factors including time patterns, "
        "global power metrics, and sub-metering readings."
    )

    # Add instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown(
        """
        1. Upload your historical data (CSV or TXT).
        2. In the sidebar, enter your estimated cost per kWh and current meter balance.
        3. Choose how many hours to forecast.
        4. Click 'Generate Forecast'.
        5. Review the forecast results, prediction summary (showing max & total predicted power), and decision support recommendations in the 'Forecast Results' tab.
        6. Explore average consumption patterns in the 'Consumption Patterns' tab.
        7. Download the detailed forecast as a CSV if needed.
        """
    )

else:
    st.error("Could not load the model or scaler. Please ensure the files are in the correct location.")
    st.info(
        "Make sure the following files exist:\n"
        "- lstm_single_household_units_model.h5\n"
        "- scaler_single_household.pkl"
    )
    
