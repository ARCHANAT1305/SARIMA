# Exp.no: 10 IMPLEMENTATION OF SARIMA MODEL
### Date:6/5/25
## AIM:
To implement SARIMA model using python.

## ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
## PROGRAM:
### NAME: ARCHANA T
### REGISTER NUMBER : 212223240013
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load and prepare data
data = pd.read_csv('powerconsumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data.asfreq('10min')  # Explicitly set frequency to avoid warning

# Select and clean the column for modeling
data = data[['PowerConsumption_Zone1']].dropna()

# Plot the time series
plt.figure(figsize=(12, 4))
plt.plot(data.index, data['PowerConsumption_Zone1'])
plt.xlabel('Date')
plt.ylabel('Power Consumption (Zone 1)')
plt.title('Power Consumption Time Series - Zone 1')
plt.tight_layout()
plt.show()

# Check stationarity using ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['PowerConsumption_Zone1'])

# Plot ACF and PACF
plot_acf(data['PowerConsumption_Zone1'].dropna(), lags=50)
plt.tight_layout()
plt.show()

plot_pacf(data['PowerConsumption_Zone1'].dropna(), lags=50)
plt.tight_layout()
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train = data['PowerConsumption_Zone1'][:train_size]
test = data['PowerConsumption_Zone1'][train_size:]

# Fit SARIMA model (You can tune the order values further)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 144), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

# Predict
predictions = sarima_result.predict(start=test.index[0], end=test.index[-1], dynamic=False)

# Evaluate
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot predictions
plt.figure(figsize=(12, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Power Consumption (Zone 1)')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.tight_layout()
plt.show()

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/76e27498-7ba5-4758-a751-a82fd38ebf81)
![image](https://github.com/user-attachments/assets/9ed72de2-11bd-4db3-8d69-058da5df6cac)

![image](https://github.com/user-attachments/assets/68554492-7d48-42fd-8343-d48d3c637301)

## RESULT:
Thus the program run successfully based on the SARIMA model.
