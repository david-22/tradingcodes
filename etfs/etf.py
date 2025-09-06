import pandas as pd 
import numpy as numpy
from matplotlib import pyplot as plt 

pd.set_option('display.max_columns', None)

df = pd.read_excel('etfdata.xlsx')
df.columns = [col.strip().lower() for col in df.columns]
df['premium'] = df['premium'].str.replace('k', '000').str.replace('$', '').str.replace('m', '0000').str.replace('.', '').str.replace(' ', '').str.replace('aa', '').str.replace('bb', '')
df['premium'] = pd.to_numeric(df['premium'], errors = 'coerce')

df_grouped = df.groupby('symbol').agg({'premium': 'sum'})
df_grouped.reset_index(inplace=True)

df_grouped.sort_values(by='premium', ascending= False, inplace=True)
df_grouped = df_grouped.rename(columns = {'symbol': 'ticker'})

print(df_grouped)
df_grouped.to_excel('newset.xlsx', index = False)