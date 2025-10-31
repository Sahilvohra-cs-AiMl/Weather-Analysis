import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1) Load dataset
fname = 'city_temperature.csv'
try:
    df = pd.read_csv(fname)
    print("Loaded", fname)
except FileNotFoundError:
    print(f"File '{fname}' not found. Please place the dataset in the notebook folder or change 'fname' to the correct path.")
    raise

# Quick look
print(df.head(), df.shape)

# 2) Basic preprocessing
if {'Year','Month','Day'}.issubset(set(df.columns)):
    df['date'] = pd.to_datetime(df[['Year','Month','Day']], errors='coerce')

possible_temp_cols = ['avg_temp', 'AvgTemperature', 'AverageTemperature', 'AvgTemp', 'temperature', 'temp', 'avg_temp_c']
temp_col = None
for c in possible_temp_cols:
    if c in df.columns:
        temp_col = c
        break

possible_date_cols = ['date', 'Date', 'dt', 'datetime']
date_col = None
for c in possible_date_cols:
    if c in df.columns:
        date_col = c
        break

if date_col is None:
    for c in df.columns:
        if df[c].dtype == 'O':
            try:
                _ = pd.to_datetime(df[c])
                date_col = c
                break
            except:
                continue

if date_col is None:
    raise ValueError("No date column found. Please ensure your dataset has a date/datetime column.")

df['date'] = pd.to_datetime(df[date_col], errors='coerce')

if temp_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        temp_col = max(numeric_cols, key=lambda c: df[c].var())
        print("Auto-selected temperature column as:", temp_col)
    else:
        raise ValueError("No suitable temperature column found. Please rename your temperature column to one of: " + ", ".join(possible_temp_cols))

df = df[['date', temp_col] + [c for c in df.columns if c not in ['date', temp_col]]]
df = df.dropna(subset=['date', temp_col]).sort_values('date').reset_index(drop=True)
df.rename(columns={temp_col: 'temperature'}, inplace=True)

print('Data range:', df['date'].min(), 'to', df['date'].max())
print(df.head())

# 3) EDA
df.set_index('date', inplace=True)
plt.figure(figsize=(12,5))
plt.plot(df['temperature'], label='Temperature')
plt.title('Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('temperature_over_time.png')
plt.show()

print("\nBasic statistics:\n", df['temperature'].describe())

monthly = df['temperature'].resample('M').mean()
plt.figure(figsize=(12,5))
plt.plot(monthly, label='Monthly mean temperature')
plt.title('Monthly Mean Temperature')
plt.savefig('monthly_mean_temperature.png')
plt.show()

# 4) Feature engineering
df_feat = df.copy()
df_feat['year'] = df_feat.index.year
df_feat['month'] = df_feat.index.month
df_feat['day'] = df_feat.index.day
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['dayofyear'] = df_feat.index.dayofyear
df_feat['temp_roll_7'] = df_feat['temperature'].rolling(7, min_periods=1).mean()
df_feat['temp_roll_30'] = df_feat['temperature'].rolling(30, min_periods=1).mean()

for lag in [1,2,3,7,14]:
    df_feat[f'lag_{lag}'] = df_feat['temperature'].shift(lag)

df_feat = df_feat.dropna().reset_index()

print(df_feat.head())

# 5) Modeling
features = ['year','month','day','dayofweek','dayofyear','temp_roll_7','temp_roll_30','lag_1','lag_2','lag_3','lag_7','lag_14']
X = df_feat[features]
y = df_feat['temperature']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print('Train size:', X_train.shape, 'Test size:', X_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# 6) Evaluation
def evaluate(y_true, y_pred, name='Model'):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} --> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

evaluate(y_test, pred_lr, 'Linear Regression')
evaluate(y_test, pred_rf, 'Random Forest')

plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, pred_lr, label='Linear Regression Pred')
plt.plot(y_test.index, pred_rf, label='Random Forest Pred')
plt.legend()
plt.title('Actual vs Predicted (Test Set)')
plt.savefig('actual_vs_predicted.png')
plt.show()

# 7) Simple 30-day forecast using Random Forest
last_row = df_feat.iloc[-1:].copy()
forecast_horizon = 30
forecast_rows = []
current = last_row.copy()

for i in range(1, forecast_horizon+1):
    new = {}
    new_date = pd.to_datetime(current['date'].iloc[0]) + pd.Timedelta(days=1)
    new['date'] = new_date
    new['year'] = new_date.year
    new['month'] = new_date.month
    new['day'] = new_date.day
    new['dayofweek'] = new_date.dayofweek
    new['dayofyear'] = new_date.dayofyear
    new['temp_roll_7'] = current['temp_roll_7'].iloc[0]
    new['temp_roll_30'] = current['temp_roll_30'].iloc[0]
    last_temp = current['temperature'].iloc[0]
    if len(forecast_rows) > 0:
        last_temp = forecast_rows[-1]['temperature']
    new['lag_1'] = last_temp
    new['lag_2'] = current['lag_1'].iloc[0] if 'lag_1' in current.columns else new['lag_1']
    new['lag_3'] = current['lag_2'].iloc[0] if 'lag_2' in current.columns else new['lag_1']
    new['lag_7'] = current['lag_7'].iloc[0] if 'lag_7' in current.columns else new['lag_1']
    new['lag_14'] = current['lag_14'].iloc[0] if 'lag_14' in current.columns else new['lag_1']
    xr = pd.DataFrame([new])
    xr = xr[features]
    pred = rf.predict(xr)[0]
    new['temperature'] = pred
    forecast_rows.append(new)
    current = pd.DataFrame([new])

forecast_df = pd.DataFrame(forecast_rows).set_index('date')
print(forecast_df.head())

# 8) Plot forecast vs recent actuals
plt.figure(figsize=(12,6))
plt.plot(df['temperature'].iloc[-90:], label='Recent Actuals')
plt.plot(forecast_df.index, forecast_df['temperature'], marker='o', label='Forecast (30 days)')
plt.title('30-Day Temperature Forecast (Simple Regression)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('forecast_plot.png')
plt.show()

# 9) Save trained models (optional)
joblib.dump(lr, 'linear_regression_model.joblib')
joblib.dump(rf, 'random_forest_model.joblib')
print("Saved models: linear_regression_model.joblib and random_forest_model.joblib")
