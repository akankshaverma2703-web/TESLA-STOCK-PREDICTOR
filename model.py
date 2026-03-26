import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ✅ Read CSV with correct delimiter
df = pd.read_csv('tesla2.csv', delimiter=',')

print("Columns:", df.columns)
print(df.head())

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['is_quarter_end'] = np.where(df['Date'].dt.month % 3 == 0, 1, 0)

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

print("✅ MODEL BAN GAYA 🚀")