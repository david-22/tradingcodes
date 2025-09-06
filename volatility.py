import yfinance as yf
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

ticker_symbols = ['BAC', 'MS', 'PLD', 'ASML', 'HWC', 'PNC', 'GS', 'PGR', 'MTB', 'MTB', 'FULT']


values = []

for ticker in ticker_symbols:

	first = yf.Ticker(ticker)
	df = first.history(period="max")  # Máximo periodo disponible hasta hoy

	df = pd.DataFrame(data=df)
	
	result = df['Close'].std()
	values.append((ticker, result))


values = pd.DataFrame(data=values)

values.columns = ['ticker', 'volatility']

values.sort_values(by='volatility', ascending=False, inplace=True)

plt.bar(values['ticker'],  values['volatility'], color = 'black')
plt.xlabel('Ticker')
plt.ylabel('Volatility')
plt.title('Volatilidad Historica por Compañia')
plt.xticks(rotation=45)
plt.show()

