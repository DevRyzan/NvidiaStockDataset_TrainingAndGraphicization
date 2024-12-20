import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import os



base_dir = os.path.dirname(os.path.abspath(__file__))  
file_path = os.path.join(base_dir, "../NVIDIA_STOCK.csv")

df = pd.read_csv(file_path)



df_cleaned = df.drop([0, 1])
df_cleaned.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')

numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].apply(pd.to_numeric, errors='coerce')

X = df_cleaned[['Open', 'High', 'Low', 'Volume']]
y = df_cleaned['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Squared Error (MSE): {mse:.4f}")
print(f" Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label="Actual Prices", marker='o')
plt.plot(y_pred, label="Predicted Prices", marker='x')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Data Point")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()