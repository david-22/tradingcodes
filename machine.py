import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

pd.set_option('display.max_columns', None)

# Leer los datos
df = pd.read_csv(r"C:\Users\David\Downloads\Kraken_OHLCVT\DASHUSD_15.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.iloc[10:].reset_index(drop=True)
df = df.sort_values(by='timestamp', ascending=False)


# Indicadores técnicos
df['return'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100

# RSI
window_length = 100
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=window_length).mean()
avg_loss = loss.rolling(window=window_length).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_crossover'] = 0
df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_crossover'] = 1
df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_crossover'] = -1

# Bandas de Bollinger
period = 100
k = 2
df['bb_middle'] = df['close'].rolling(window=period).mean()
df['bb_std'] = df['close'].rolling(window=period).std()
df['bb_upper'] = df['bb_middle'] + (k * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (k * df['bb_std'])

# Momentum
momentum_period = 10
df['momentum'] = df['close'] - df['close'].shift(momentum_period)

# ATR
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['true_range'].rolling(window=14).mean()
df.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

# Eliminar filas con NaN y columnas innecesarias
df.dropna(subset=['rsi', 'bb_lower', 'bb_upper', 'bb_std', 'bb_middle'], inplace=True)
df.drop(columns=['timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)




# Variable objetivo
df['direction'] = np.where(df['return'] > 0, 1, 0)

# Separar conjunto de entrenamiento y validación
df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)
X_train = df_train.drop(['direction', 'return'], axis=1)
y_train = df_train['direction']
X_valid = df_valid.drop(['direction', 'return'], axis=1)
y_valid = df_valid['direction']

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Entrenar modelo
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Predicciones y evaluación
y_proba = model.predict_proba(X_valid_scaled)[:, 1]
threshold = 0.6
y_pred_custom = (y_proba >= threshold).astype(int)

precision = precision_score(y_valid, y_pred_custom)
recall = recall_score(y_valid, y_pred_custom)
f1 = f1_score(y_valid, y_pred_custom)
conf_matrix = confusion_matrix(y_valid, y_pred_custom)

print(f"Threshold = {threshold}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nMatriz de confusión:")
print(conf_matrix)

# Guardar modelo, scaler y columnas
features_columns = list(X_train.columns)
joblib.dump(model, r'C:\Users\David\Downloads\Kraken_OHLCVT\modelo_logistico_dash.pkl')
joblib.dump(scaler, r'C:\Users\David\Downloads\Kraken_OHLCVT\scaler_dash.pkl')
joblib.dump(features_columns, r'C:\Users\David\Downloads\Kraken_OHLCVT\features_columns.pkl')
