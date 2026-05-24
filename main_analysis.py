"""
Digital Twin IoT Power Prediction - Complete Data Mining Pipeline
Author: [Nama Mahasiswa]
Date: 2026

Dataset: sensor_data_complete.xlsx
Target: Prediksi Konsumsi Daya Listrik (Daya in Watt)
Models: XGBoost, Random Forest, LSTM
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("DIGITAL TWIN IoT POWER PREDICTION - DATA MINING PIPELINE")
print("=" * 70)
print(f"Started at: {datetime.now()}")
print()

# ============================================================================
# 2. DATA LOADING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: DATA LOADING")
print("=" * 70)

# Load dataset
df = pd.read_excel('dataset sensor_data_complete.xlsx')

print(f"✓ Dataset loaded successfully!")
print(f"  - Total records: {len(df):,}")
print(f"  - Total columns: {len(df.columns)}")
print(f"  - Columns: {df.columns.tolist()}")
print()

# Display first few rows
print("First 5 rows:")
print(df.head())

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 3.1 Data Types and Info
print("\n3.1 Data Types:")
print(df.dtypes)

# 3.2 Missing Values
print("\n3.2 Missing Values:")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

# 3.3 Statistical Summary
print("\n3.3 Statistical Summary:")
print(df.describe())

# 3.4 Convert Timestamp to Datetime
df['Timestamp (WIB)'] = pd.to_datetime(df['Timestamp (WIB)'])
df['Timestamp (UTC)'] = pd.to_datetime(df['Timestamp (UTC)'])

# 3.5 Time Range Analysis
print("\n3.4 Time Range Analysis:")
print(f"  - Start: {df['Timestamp (WIB)'].min()}")
print(f"  - End: {df['Timestamp (WIB)'].max()}")
print(f"  - Duration: {df['Timestamp (WIB)'].max() - df['Timestamp (WIB)'].min()}")

# 3.6 Correlation Analysis
print("\n3.5 Correlation Analysis:")
numeric_cols = ['Suhu (°C)', 'Kelembaban (%)', 'Tegangan (V)', 'Arus (A)', 'Daya (W)', 'Jumlah Orang']
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix.round(3))

# 3.7 Create Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis - IoT Sensor Data', fontsize=16)

# Temperature Distribution
axes[0, 0].hist(df['Suhu (°C)'], bins=50, color='coral', edgecolor='black')
axes[0, 0].set_title('Temperature Distribution')
axes[0, 0].set_xlabel('Suhu (°C)')

# Humidity Distribution
axes[0, 1].hist(df['Kelembaban (%)'], bins=50, color='skyblue', edgecolor='black')
axes[0, 1].set_title('Humidity Distribution')
axes[0, 1].set_xlabel('Kelembaban (%)')

# Voltage Distribution
axes[0, 2].hist(df['Tegangan (V)'], bins=50, color='gold', edgecolor='black')
axes[0, 2].set_title('Voltage Distribution')
axes[0, 2].set_xlabel('Tegangan (V)')

# Power Distribution
axes[1, 0].hist(df['Daya (W)'], bins=50, color='green', edgecolor='black')
axes[1, 0].set_title('Power Distribution')
axes[1, 0].set_xlabel('Daya (W)')

# Current Distribution
axes[1, 1].hist(df['Arus (A)'], bins=50, color='purple', edgecolor='black')
axes[1, 1].set_title('Current Distribution')
axes[1, 1].set_xlabel('Arus (A)')

# Correlation Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ EDA visualization saved as 'eda_visualization.png'")
plt.close()

# 3.8 Time Series Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Sample data for plotting (to avoid too many points)
sample_df = df.iloc[::100]  # Every 100th row

# Power over time
axes[0].plot(sample_df['Timestamp (WIB)'], sample_df['Daya (W)'], 'g-', linewidth=0.8)
axes[0].set_title('Power Consumption Over Time')
axes[0].set_ylabel('Daya (W)')
axes[0].grid(True, alpha=0.3)

# Temperature over time
axes[1].plot(sample_df['Timestamp (WIB)'], sample_df['Suhu (°C)'], 'r-', linewidth=0.8)
axes[1].set_title('Temperature Over Time')
axes[1].set_ylabel('Suhu (°C)')
axes[1].grid(True, alpha=0.3)

# Humidity over time
axes[2].plot(sample_df['Timestamp (WIB)'], sample_df['Kelembaban (%)'], 'b-', linewidth=0.8)
axes[2].set_title('Humidity Over Time')
axes[2].set_ylabel('Kelembaban (%)')
axes[2].set_xlabel('Timestamp')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_plot.png', dpi=150, bbox_inches='tight')
print("✓ Time series visualization saved as 'time_series_plot.png'")
plt.close()

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: DATA PREPROCESSING")
print("=" * 70)

# 4.1 Remove columns we don't need
df_clean = df.drop(['No', 'Timestamp (UTC)', 'Device ID'], axis=1)

# 4.2 Handle Missing Values in 'Jumlah Orang'
print("\n3.1 Handling Missing Values:")
print(f"  - 'Jumlah Orang' has {df_clean['Jumlah Orang'].isnull().sum():,} missing values ({df_clean['Jumlah Orang'].isnull().sum()/len(df_clean)*100:.1f}%)")

# Option 1: Drop rows with missing Jumlah Orang (for better accuracy)
# Option 2: Fill with median/mode
# We choose to drop because 72% missing is too high
df_clean = df_clean.dropna(subset=['Daya (W)', 'Suhu (°C)', 'Kelembaban (%)', 'Tegangan (V)', 'Arus (A)'])

print(f"  - After cleaning: {len(df_clean):,} records")

# 4.3 Feature Engineering
print("\n3.2 Feature Engineering:")

# Extract time-based features
df_clean['Hour'] = df_clean['Timestamp (WIB)'].dt.hour
df_clean['DayOfWeek'] = df_clean['Timestamp (WIB)'].dt.dayofweek
df_clean['Day'] = df_clean['Timestamp (WIB)'].dt.day
df_clean['Month'] = df_clean['Timestamp (WIB)'].dt.month

# Create time period categories
def categorize_time(hour):
    if 6 <= hour < 10:
        return 'Morning'
    elif 10 <= hour < 14:
        return 'Midday'
    elif 14 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df_clean['TimePeriod'] = df_clean['Hour'].apply(categorize_time)

# Create interaction features
df_clean['Suhu_Kelembaban'] = df_clean['Suhu (°C)'] * df_clean['Kelembaban (%)']
df_clean['Tegangan_Arus'] = df_clean['Tegangan (V)'] * df_clean['Arus (A)']

# Rolling statistics (using 5 minute window ~ 100 records at 3-sec intervals)
df_clean = df_clean.sort_values('Timestamp (WIB)').reset_index(drop=True)
df_clean['Daya_MA_5'] = df_clean['Daya (W)'].rolling(window=100, min_periods=1).mean()
df_clean['Daya_MA_15'] = df_clean['Daya (W)'].rolling(window=300, min_periods=1).mean()
df_clean['Suhu_MA_5'] = df_clean['Suhu (°C)'].rolling(window=100, min_periods=1).mean()

print("  - Extracted features: Hour, DayOfWeek, Day, Month, TimePeriod")
print("  - Created interaction features: Suhu_Kelembaban, Tegangan_Arus")
print("  - Created rolling statistics: Daya_MA_5, Daya_MA_15, Suhu_MA_5")

# 4.4 Encode categorical variables
df_clean = pd.get_dummies(df_clean, columns=['TimePeriod'], drop_first=True)

# Ensure all time period columns exist
for col in ['TimePeriod_Morning', 'TimePeriod_Midday', 'TimePeriod_Afternoon', 'TimePeriod_Evening']:
    if col not in df_clean.columns:
        df_clean[col] = 0

# 4.5 Define Features and Target
feature_columns = [
    'Suhu (°C)', 'Kelembaban (%)', 'Tegangan (V)', 'Arus (A)', 'Jumlah Orang',
    'Hour', 'DayOfWeek', 'Day',
    'Suhu_Kelembaban', 'Tegangan_Arus',
    'Daya_MA_5', 'Daya_MA_15', 'Suhu_MA_5',
    'TimePeriod_Morning', 'TimePeriod_Midday', 'TimePeriod_Afternoon', 'TimePeriod_Evening'
]

X = df_clean[feature_columns]
y = df_clean['Daya (W)']

print(f"\n  - Features: {len(feature_columns)}")
print(f"  - Target: Daya (W)")
print(f"  - Total samples: {len(X):,}")

# 4.6 Train-Test Split (Time Series - chronological split, NO shuffle!)
print("\n3.3 Train-Test Split (Chronological):")
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"  - Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# 4.7 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  - Applied StandardScaler for feature normalization")

# ============================================================================
# 5. MODELING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MODELING")
print("=" * 70)

# Function to evaluate model
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n  {model_name}:")
    print(f"    - RMSE:  {rmse:.4f} W")
    print(f"    - MAE:   {mae:.4f} W")
    print(f"    - R²:    {r2:.4f}")
    print(f"    - MAPE:  {mape:.2f}%")

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

results = []

# 5.1 Linear Regression (Baseline)
print("\n4.1 Linear Regression (Baseline)")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
results.append(evaluate_model(y_test, lr_pred, 'Linear Regression'))

# 5.2 Random Forest
print("\n4.2 Random Forest Regressor")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # RF doesn't need scaling
rf_pred = rf_model.predict(X_test)
results.append(evaluate_model(y_test, rf_pred, 'Random Forest'))

# Feature Importance (Random Forest)
print("\n  Feature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# 5.3 XGBoost (Primary Model)
print("\n4.3 XGBoost Regressor (Primary Model)")
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results.append(evaluate_model(y_test, xgb_pred, 'XGBoost'))

# Feature Importance (XGBoost)
print("\n  Feature Importance (XGBoost):")
xgb_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance.head(10).to_string(index=False))

# ============================================================================
# 6. DEEP LEARNING - LSTM (Time Series)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: DEEP LEARNING - LSTM")
print("=" * 70)

lstm_pred = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Prepare data for LSTM (needs 3D input: samples, timesteps, features)
    lstm_features = ['Suhu (°C)', 'Kelembaban (%)', 'Tegangan (V)', 'Arus (A)', 'Daya (W)']
    lstm_data = df_clean[lstm_features].values

    # Normalize data
    lstm_scaler = MinMaxScaler()
    lstm_data_scaled = lstm_scaler.fit_transform(lstm_data)

    # Create sequences
    def create_sequences(data, seq_length, target_col_idx):
        X_seq, y_seq = [], []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i+seq_length])
            y_seq.append(data[i+seq_length, target_col_idx])
        return np.array(X_seq), np.array(y_seq)

    SEQ_LENGTH = 60  # ~3 minutes of data
    X_seq, y_seq = create_sequences(lstm_data_scaled, SEQ_LENGTH, 4)  # 4 is Daya index

    # Split
    train_seq = int(len(X_seq) * 0.8)
    X_train_lstm = X_seq[:train_seq]
    X_test_lstm = X_seq[train_seq:]
    y_train_lstm = y_seq[:train_seq]
    y_test_lstm = y_seq[train_seq:]

    print(f"\n  - Sequence length: {SEQ_LENGTH}")
    print(f"  - Training sequences: {len(X_train_lstm):,}")
    print(f"  - Test sequences: {len(X_test_lstm):,}")

    # Build LSTM model
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 5)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("\n  LSTM Model Architecture:")
    lstm_model.summary()

    # Train LSTM
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\n  Training LSTM model...")
    history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Predict
    lstm_pred_scaled = lstm_model.predict(X_test_lstm)

    # Inverse transform predictions
    # Create dummy arrays with same shape for inverse transform
    dummy_pred = np.zeros((len(lstm_pred_scaled), 5))
    dummy_pred[:, 4] = lstm_pred_scaled.flatten()
    lstm_pred = lstm_scaler.inverse_transform(dummy_pred)[:, 4]

    dummy_actual = np.zeros((len(y_test_lstm), 5))
    dummy_actual[:, 4] = y_test_lstm
    y_test_actual = lstm_scaler.inverse_transform(dummy_actual)[:, 4]

    results.append(evaluate_model(y_test_actual, lstm_pred, 'LSTM'))

except ImportError:
    print("\n  TensorFlow not available. Skipping LSTM model.")
    print("  To install: pip install tensorflow")

# ============================================================================
# 7. MODEL COMPARISON & EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: MODEL COMPARISON & EVALUATION")
print("=" * 70)

# Create comparison dataframe
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['rmse'].idxmin()
best_model_name = results_df.loc[best_model_idx, 'model']
print(f"\n✓ Best Model: {best_model_name} (lowest RMSE: {results_df.loc[best_model_idx, 'rmse']:.4f} W)")

# Visualization: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot comparison
x_pos = np.arange(len(results_df))
axes[0].bar(x_pos, results_df['rmse'], color=['blue', 'green', 'orange', 'red'][:len(results_df)])
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(results_df['model'], rotation=45)
axes[0].set_ylabel('RMSE (W)')
axes[0].set_title('Model Comparison - RMSE')
axes[0].grid(True, alpha=0.3)

# Actual vs Predicted plot (using best model predictions)
if best_model_name == 'XGBoost':
    best_pred = xgb_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
elif best_model_name == 'LSTM':
    best_pred = lstm_pred
else:
    best_pred = lr_pred

axes[1].scatter(y_test.values[::10], best_pred[::10], alpha=0.5, s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Power (W)')
axes[1].set_ylabel('Predicted Power (W)')
axes[1].set_title(f'Actual vs Predicted - {best_model_name}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Model comparison visualization saved as 'model_comparison.png'")
plt.close()

# ============================================================================
# 8. SAVE BEST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVE MODEL FOR DEPLOYMENT")
print("=" * 70)

import joblib

# Save XGBoost model (our primary model)
model_path = 'xgboost_power_prediction_model.joblib'
joblib.dump(xgb_model, model_path)
print(f"✓ XGBoost model saved as '{model_path}'")

# Save scaler
scaler_path = 'feature_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved as '{scaler_path}'")

# Save feature columns
joblib.dump(feature_columns, 'feature_columns.joblib')
print(f"✓ Feature columns saved as 'feature_columns.joblib'")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ Pipeline completed successfully!

Dataset Information:
  - Total Records: {len(df):,}
  - Features Used: {len(feature_columns)}
  - Training Samples: {len(X_train):,}
  - Test Samples: {len(X_test):,}

Best Model: {best_model_name}
  - RMSE: {results_df.loc[best_model_idx, 'rmse']:.4f} W
  - MAE: {results_df.loc[best_model_idx, 'mae']:.4f} W
  - R²: {results_df.loc[best_model_idx, 'r2']:.4f}
  - MAPE: {results_df.loc[best_model_idx, 'mape']:.2f}%

Files Generated:
  - eda_visualization.png
  - time_series_plot.png
  - model_comparison.png
  - xgboost_power_prediction_model.joblib
  - feature_scaler.joblib
  - feature_columns.joblib

Next Steps:
  1. Run 'streamlit_app.py' for deployment
  2. Create Digital Twin dashboard
  3. Implement real-time prediction

Completed at: {datetime.now()}
""")
print("=" * 70)