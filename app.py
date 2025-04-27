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
MODEL_PATH = 'lstm_single_household_units_model.h5'
SCALER_PATH = 'scaler_single_household.pkl'
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
    # Try relative path first
    if os.path.exists(model_path):
        abs_model_path = model_path
    else:
        # If relative path fails, try the hardcoded local dev path
        local_dev_path = 'C:/Users/ebina/Documents/Personal/NYSC/Data_Science_Advanced/Group_2_Projects/electricity_consumption/' + model_path
        if os.path.exists(local_dev_path):
             abs_model_path = local_dev_path
        else:
            st.error(f"Model file not found. Looked for '{model_path}' (relative) and '{local_dev_path}' (absolute). Please ensure the file is in the correct location.")
            return None

    try:
        model = tf.keras.models.load_model(abs_model_path, custom_objects={'mse': MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"Error loading model from {abs_model_path}: {e}")
        return None

# Function to load the saved scaler
@st.cache_resource # Cache the scaler
def load_scaler(scaler_path):
    # Try relative path first
    if os.path.exists(scaler_path):
        abs_scaler_path = scaler_path
    else:
        # If relative path fails, try the hardcoded local dev path
        local_dev_path = 'C:/Users/ebina/Documents/Personal/NYSC/Data_Science_Advanced/Group_2_Projects/electricity_consumption/' + scaler_path
        if os.path.exists(local_dev_path):
             abs_scaler_path = local_dev_path
        else:
            st.error(f"Scaler file not found. Looked for '{scaler_path}' (relative) and '{local_dev_path}' (absolute). Please ensure the file is in the correct location.")
            return None

    try:
        with open(abs_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from {abs_scaler_path}: {e}")
        return None

# Function to preprocess new data for prediction
# Fix: Removed target_feature from args, use global TARGET_FEATURE
def preprocess_data(df, scaler, lookback_period, lstm_features):
    # Ensure data is sorted by time
    df = df.sort_values(by='time').copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Create derived time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    # Add other time features if your model used them (e.g., dayofyear, month)

    # Select the features the LSTM was trained on
    # Check for required original columns vs derived columns
    required_original_cols = [f for f in lstm_features if f not in ['hour_of_day', 'day_of_week']]
    missing_original_cols = [col for col in required_original_cols if col not in df.columns]

    if missing_original_cols:
         st.error(f"Missing required original columns in the uploaded data: {', '.join(missing_original_cols)}")
         return None, None, None

    # Now select the features including the newly created ones
    try:
        data_for_lstm = df[lstm_features]
    except KeyError as e:
         # This should ideally not happen if missing_original_cols check passed, but as a safeguard
        st.error(f"Internal Error: Could not select all LSTM features after creating time features. Missing: {e}")
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
        st.error(f"Error during scaling. Please check if your uploaded data columns and their order match the scaler's expected features: {e}")
        return None, None, None

    # Reshape for LSTM input [n_samples, lookback, n_features]
    # We only have one sample (the last lookback_period), so n_samples = 1
    X = scaled_data.reshape(1, lookback_period, len(lstm_features))

    # Get the last timestamp for plotting reference
    last_timestamp = recent_data.index[-1]

    # Return the full processed dataframe slice for plotting history
    return X, last_timestamp, df # Return df to plot history


# Function to inverse transform predictions (for the target feature)
# Fix: Removed target_feature from args, use global TARGET_FEATURE
def inverse_transform_prediction(scaled_prediction, scaler, lstm_features):
    # The scaler was fitted on N_LSTM_FEATURES, but we only predicted 1 (target_feature).
    # To inverse transform correctly, we need to create a dummy array with the predicted value
    # in the correct column position and other columns filled with a placeholder (like the mean or 0).
    try:
        target_col_index = lstm_features.index(TARGET_FEATURE) # Use global TARGET_FEATURE
    except ValueError:
        st.error(f"Internal error: Target feature '{TARGET_FEATURE}' not found in the LSTM feature list.")
        return None

    # Create a dummy array with the shape expected by the scaler
    dummy_array = np.zeros((scaled_prediction.shape[0], len(lstm_features)))

    # Place the scaled prediction into the target feature's column
    dummy_array[:, target_col_index] = scaled_prediction.flatten()

    # Inverse transform the dummy array
    # This transforms the entire row, but we only care about the target column result
    original_scale_data = scaler.inverse_transform(dummy_array)

    # The target prediction is now in the correct column of the inverse-transformed array
    predicted_target = original_scale_data[:, target_col_index]

    return predicted_target


# --- DSS Logic Functions ---

# Function to provide budget recommendations based on predicted consumption and balance
def get_budget_recommendation(total_predicted_units, current_balance_naira, cost_per_unit_naira):
    recommendations = []
    
    if cost_per_unit_naira <= 0:
        recommendations.append("Please provide a valid cost per unit to get budget recommendations.")
        return recommendations

    estimated_cost_naira = total_predicted_units * cost_per_unit_naira

    recommendations.append(f"Estimated cost for the predicted usage period ({len(predicted_values)} hours): **₦{estimated_cost_naira:,.2f}**") # Use len(predicted_values) for clarity

    # Calculate the value of current balance in Units
    units_in_balance = current_balance_naira / cost_per_unit_naira

    if total_predicted_units > units_in_balance:
        units_needed_strictly = total_predicted_units - units_in_balance
        # Recommend buying a buffer, e.g., 10-20% more than strictly needed
        recommended_units_to_buy = units_needed_strictly * 1.15 # Recommend 15% buffer

        recommendations.append(f"⚠️ Your predicted usage (**{total_predicted_units:.2f} Units**) will likely **exceed** the units equivalent of your current balance (**{units_in_balance:.2f} Units**).")
        recommendations.append(f"**Recommendation:** Consider buying approximately **{recommended_units_to_buy:.2f} Units** (₦{recommended_units_to_buy * cost_per_unit_naira:,.2f}) to cover the predicted usage plus a buffer.")

    elif total_predicted_units <= units_in_balance and total_predicted_units > units_in_balance * 0.8: # Usage is within 80-100% of balance equivalent units
         recommendations.append(f"🗸 Your current balance (**{units_in_balance:.2f} Units**) appears sufficient for the predicted usage (**{total_predicted_units:.2f} Units**), but you are close to your limit.")
         recommendations.append("Consider topping up soon if you anticipate higher future usage or for peace of mind.")
    else: # Usage is well below the balance equivalent units
         recommendations.append(f"✅ Your current balance (**{units_in_balance:.2f} Units**) appears well above the predicted consumption (**{total_predicted_units:.2f} Units**) for this period.")
         recommendations.append(f"You're doing good!!")

    return recommendations

# Function to identify and warn about predicted peak usage times
def get_peak_usage_warning(prediction_df, peak_threshold_multiplier=1.25): # Increased multiplier slightly
    recommendations = []
    predicted_units = prediction_df[f'Predicted_{TARGET_FEATURE}']

    if predicted_units.empty:
        return ["No prediction data available to identify peaks."]

    # Calculate the average predicted consumption
    average_predicted = predicted_units.mean()
    # Define a threshold for peak hours (e.g., 25% above average)
    peak_threshold = average_predicted * peak_threshold_multiplier

    # Find hours where predicted consumption exceeds the threshold
    peak_hours_df = prediction_df[predicted_units > peak_threshold]

    if not peak_hours_df.empty:
        # Sort peak hours by predicted value (highest first)
        peak_hours_df = peak_hours_df.sort_values(f'Predicted_{TARGET_FEATURE}', ascending=False)

        peak_times = peak_hours_df.index.strftime('%I:%M %p on %A, %b %d').tolist() # More readable time format
        recommendations.append(f"🔥 **Predicted High Usage Peaks** ({peak_threshold:.2f}+ {TARGET_FEATURE}/hour) at the following times:")
        # Limit the number of peak times displayed for brevity
        display_limit = 7 # Show up to 7 peak times
        for i, time_str in enumerate(peak_times[:display_limit]):
            rec_units = peak_hours_df[f'Predicted_{TARGET_FEATURE}'].iloc[i]
            recommendations.append(f"- {time_str}: **{rec_units:.2f} Units**")
        if len(peak_times) > display_limit:
             recommendations.append(f"...and {len(peak_times) - display_limit} more hours with high predicted usage.")

        recommendations.append(f"\n**Actionable Tip:** Consider reducing non-essential appliance usage (like running multiple high-power devices simultaneously) or shifting high-load tasks (like laundry, heavy ironing, prolonged AC use) during these highlighted peak periods to potentially manage cost and load.")
    else:
        recommendations.append("👍 No significant peak consumption predicted above the set threshold for this period.")

    return recommendations

# --- Streamlit App Layout ---
st.title("Household Electricity Units Estimator")

st.write("""
Upload your historical household data (CSV format) to get predictions.
and decision support recommendations for future hours using a pre-trained LSTM model.
""")

# --- User Input for DSS Context ---
st.sidebar.header("Your Inputs for DSS")

cost_per_unit = st.sidebar.number_input(
    "Estimated Cost Per Unit (₦)",
    min_value=0.0,
    value=50.0, # Example average cost per unit in Nigeria, adjust as needed
    step=0.1, # Allow decimal input
    format="%.2f",
    help="Enter the approximate cost in Naira for 1 unit (kWh) of electricity."
)

current_balance = st.sidebar.number_input(
    "Current Meter Balance (₦)",
    min_value=0.0,
    value=5000.0, # Example balance
    step=100.0,
    format="%.2f",
    help="Enter your current balance on your prepaid meter in Naira."
)

# Add a simple goal selection
goal_optimize_cost = st.sidebar.checkbox("Goal: Optimize Electricity Cost", value=True)
# Future: add more goals like minimize consumption, reliability, etc.

st.sidebar.markdown("---")


# --- Load Model and Scaler ---
# Update paths here if you are still testing locally and not using relative paths
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# Display loading status messages
if model is None or scaler is None:
    # The load_model/load_scaler functions already display an error message
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

        st.write("**Data Preview:**")
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
        if st.button(f"Run Prediction and DSS"):
            # Fix: Removed target_feature from preprocess_data call
            X, last_timestamp, df_processed = preprocess_data(
                df.copy(), scaler, LOOKBACK_PERIOD, LSTM_INPUT_FEATURES
            )

            if X is not None and last_timestamp is not None and df_processed is not None:
                 st.subheader(f"Prediction for the next {n_predict_steps} hours:") # More precise start time

                 predicted_values = []
                 current_input_sequence = X # Start with the preprocessed last lookback_period data
                 current_timestamp = last_timestamp # Start timestamp for predictions

                 # Get the last known values of all input features to use as base for future steps
                 # We need to handle potential errors if some columns are missing after preprocess
                 try:
                      # Ensure df_processed has all LSTM_INPUT_FEATURES before trying to get last values
                      if not all(item in df_processed.columns for item in LSTM_INPUT_FEATURES):
                           raise KeyError("Processed data missing some LSTM input features.")
                      last_known_features = df_processed[LSTM_INPUT_FEATURES].iloc[-1].to_dict()
                 except KeyError as e:
                      st.error(f"Could not get last known features required for prediction loop. Ensure all required columns were in the uploaded data. Details: {e}")
                      last_known_features = None # Indicate error state


                 if last_known_features is not None:
                      # Multi-step prediction loop
                      # Keep track of predictions with timestamps for the DSS logic
                      prediction_data_points = []

                      for i in range(n_predict_steps):
                          # Make prediction for the next step
                          # The model expects input shape (1, lookback, N_LSTM_FEATURES)
                          scaled_next_prediction = model.predict(current_input_sequence, verbose=0)

                          # Inverse transform the prediction (only the target feature)
                          # Fix: Removed target_feature from inverse_transform_prediction call
                          next_prediction_target = inverse_transform_prediction(
                              scaled_next_prediction, scaler, LSTM_INPUT_FEATURES
                          )
                          if next_prediction_target is None: # Handle error from inverse_transform
                               break # Stop prediction loop on error

                          predicted_unit_value = next_prediction_target[0]
                          predicted_values.append(predicted_unit_value) # Store the single predicted value

                          # Add the prediction and its timestamp to the list for creating prediction_df later
                          prediction_data_points.append({'time': current_timestamp + pd.Timedelta(hours=1),
                                                         f'Predicted_{TARGET_FEATURE}': predicted_unit_value})


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
                          next_step_features_dict[TARGET_FEATURE] = predicted_unit_value # Use the model's *new* prediction for the target

                          # For other features that change (temperature, appliance status, etc.),
                          # we are using the last known values as a simplification.
                          # In a real DSS, you might use forecasted weather or more sophisticated methods.

                          new_point_features_df = pd.DataFrame([next_step_features_dict]) # Create a tiny DataFrame
                          new_point_features_df = new_point_features_df[LSTM_INPUT_FEATURES] # Ensure column order matches training

                          # 4. Scale the new point using the same scaler
                          try:
                              scaled_new_point = scaler.transform(new_point_features_df)
                          except Exception as e:
                               st.error(f"Error scaling next prediction input: {e}")
                               break # Stop prediction loop on error

                          # 5. Append the scaled new point to the sequence and remove the oldest point
                          # current_input_sequence shape is (1, lookback, features)
                          # scaled_new_point shape is (1, features) -> needs to be (1, 1, features)
                          scaled_new_point_reshaped = scaled_new_point.reshape(1, 1, N_LSTM_FEATURES)

                          # Remove the first element (oldest time step) and append the new one
                          current_input_sequence = np.append(current_input_sequence[:, 1:, :], scaled_new_point_reshaped, axis=1)


                      # --- Create prediction_df for DSS and Visualization ---
                      if prediction_data_points: # Check if prediction loop completed successfully and produced values
                          prediction_df = pd.DataFrame(prediction_data_points).set_index('time')

                          # --- DSS Output Generation ---
                          st.subheader("Decision Support Recommendations")

                          total_predicted_units = prediction_df[f'Predicted_{TARGET_FEATURE}'].sum()
                          st.info(f"Total Predicted {TARGET_FEATURE} over the next {n_predict_steps} hours: **{total_predicted_units:.2f} Units**")

                          # Budget Recommendation
                          budget_recs = get_budget_recommendation(total_predicted_units, current_balance, cost_per_unit)
                          for rec in budget_recs:
                              if "⚠️" in rec:
                                  st.warning(rec)
                              elif "🗸" in rec:
                                  st.success(rec)
                              else:
                                  st.write(rec)

                          st.markdown("---")

                          # Peak Usage Warning (only if goal is cost optimization or consumption management)
                          if goal_optimize_cost: # Only show peak warnings if optimizing cost is a goal
                              peak_recs = get_peak_usage_warning(prediction_df)
                              for rec in peak_recs:
                                  if "🔥" in rec:
                                      st.error(rec)
                                  elif "Actionable Tip:" in rec:
                                       st.markdown(rec) # Use markdown for bold tip
                                  else:
                                      st.write(rec)

                              st.markdown("---") # Separator


                          # --- Visualization ---
                          st.subheader("Historical Data and Future Prediction")
                          fig, ax = plt.subplots(figsize=(15, 6))

                          # Plot historical data (last ~LOOKBACK_PERIOD * 2 rows for context)
                          # Ensure there's enough historical data to plot
                          if len(df_processed) >= LOOKBACK_PERIOD * 2:
                             historical_plot_df = df_processed[TARGET_FEATURE].tail(LOOKBACK_PERIOD * 2)
                          else:
                             historical_plot_df = df_processed[TARGET_FEATURE] # Plot all available historical data
                             st.info(f"Showing all available historical data ({len(historical_plot_df)} rows) as it's less than {LOOKBACK_PERIOD * 2} for context.")


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

                      else: # This handles cases where the prediction loop broke or returned empty predicted_values
                          st.warning("Prediction could not be completed.")

            # This else block handles failures in preprocess_data
            else:
                 st.warning("Preprocessing failed. Please check your data file.")


    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        import traceback
        st.error(traceback.format_exc()) # Show traceback for debugging

else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.write("Note: The multi-step prediction in this demo makes a simplifying assumption about future input features (like temperature, appliance status) by using the last known values.")
st.write("DSS Recommendations are based on basic rules and user-provided cost/balance information.")