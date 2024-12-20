import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd


base_dir = os.path.dirname(os.path.abspath(__file__))  
file_path = os.path.join(base_dir, "../NVIDIA_STOCK.csv")

df = pd.read_csv(file_path)

df_cleaned = df.drop([0, 1])
df_cleaned.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')

numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(14, 7))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Close Price', linestyle='-', marker='.')
plt.plot(df_cleaned['Date'], df_cleaned['High'], label='High Price', linestyle='--', marker='.')
plt.plot(df_cleaned['Date'], df_cleaned['Low'], label='Low Price', linestyle=':', marker='.')
plt.plot(df_cleaned['Date'], df_cleaned['Open'], label='Open Price', linestyle='-.', marker='.')

plt.title("NVIDIA Stock Price Comparing")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()