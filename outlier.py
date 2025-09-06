import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# Cargar datos
d = pd.read_csv('1.csv')
df = pd.DataFrame(data=d)

# Limpiar nombres de columnas
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Eliminar columnas unnamed
df = df.loc[:, ~df.columns.str.startswith('unnamed')]

# Renombrar columnas clave
df.rename(columns={
    'last_close': 'price',
    'last_yr`s_eps_(f0)_before_nri': 'eps0',
    'f1_consensus_est.': 'eps1',
    'f2_consensus_est.': 'eps2',
    'market_cap_(mil)': 'marketcap'
}, inplace=True)

# Filtrado y limpieza
df = df[df['exchange'] != 'OTC']
df.dropna(subset=['eps0', 'eps1', 'eps2', 'company_name'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Cálculo de crecimiento EPS
def calculate_eg1_column(row):
    if (row['eps0'] < 0 and row['eps1'] > 0):
        return 100.0
    elif (row['eps0'] > 0 and row['eps1'] < 0):
        return -100.0
    elif row['eps0'] == 0:
        return np.nan
    else:
        return (row['eps1'] - row['eps0']) / abs(row['eps0']) * 100

def calculate_eg2_column(row):
    if row['eps1'] < 0 and row['eps2'] > 0:
        return 100.0
    elif row['eps1'] > 0 and row['eps2'] < 0:
        return -100.0
    elif row['eps1'] == 0:
        return np.nan
    else:
        return (row['eps2'] - row['eps1']) / abs(row['eps1']) * 100

df['EG1'] = df.apply(calculate_eg1_column, axis=1)
df['EG2'] = df.apply(calculate_eg2_column, axis=1)

# Cálculo de PE y PEG
df['PE1'] = df['price'] / df['eps1']
df['PE2'] = df['price'] / df['eps2']
df['PEG1'] = (df['PE1'] / (df['EG1'] * 100)) * 100
df['PEG2'] = (df['PE2'] / (df['EG2'] * 100)) * 100

# Clasificación por capitalización
def classify_market_cap(market_cap):
    if 1000 <= market_cap <= 10000:
        return 'Low Cap'
    elif 10000 < market_cap <= 30000:
        return 'Middle Cap'
    else:
        return 'High Cap'

df['market_cap_category'] = df['marketcap'].apply(classify_market_cap)

df = df.replace([np.inf, -np.inf], np.nan)
df.fillna(0, inplace=True)

#df.to_excel('PE data filtered.xlsx', index=None)

# Industria específica a analizar
industry_look = 'Electronics - Miscellaneous Products'
industry_specified = df[df['industry'] == industry_look]

industry_concat = industry_specified
industry_concat.fillna('', inplace=True)

# Outliers EG1
Q1_eg1 = industry_concat['EG1'].quantile(0.25)
Q3_eg1 = industry_concat['EG1'].quantile(0.75)
IQR1 = Q3_eg1 - Q1_eg1
limite_infierior_eg1 = Q1_eg1 - 1.5 * IQR1
limite_superior_eg1 = Q3_eg1 + 1.5 * IQR1
outliers = industry_concat[(industry_concat['EG1'] < limite_infierior_eg1) | (industry_concat['EG1'] > limite_superior_eg1)]

# Outliers EG2
Q1_eg2 = industry_concat['EG2'].quantile(0.25)
Q3_eg2 = industry_concat['EG2'].quantile(0.75)
IQR2 = Q3_eg2 - Q1_eg2
limite_infierior_eg2 = Q1_eg2 - 1.5 * IQR2
limite_superior_eg2 = Q3_eg2 + 1.5 * IQR2
outliers_eg2 = industry_concat[(industry_concat['EG2'] < limite_infierior_eg2) | (industry_concat['EG2'] > limite_superior_eg2)]

# Outliers PE1
Q1_pe1 = industry_concat['PE1'].quantile(0.25)
Q3_pe1 = industry_concat['PE1'].quantile(0.75)
IQR3 = Q3_pe1 - Q1_pe1
limite_infierior_pe1 = Q1_pe1 - 1.5 * IQR3
limite_superior_pe1 = Q3_pe1 + 1.5 * IQR3
outliers_pe1 = industry_concat[(industry_concat['PE1'] < limite_infierior_pe1) | (industry_concat['PE1'] > limite_superior_pe1)]

# Outliers PE2
Q1_pe2 = industry_concat['PE2'].quantile(0.25)
Q3_pe2 = industry_concat['PE2'].quantile(0.75)
IQR4 = Q3_pe2 - Q1_pe2
limite_infierior_pe2 = Q1_pe2 - 1.5 * IQR4
limite_superior_pe2 = Q3_pe2 + 1.5 * IQR4
outliers_pe2 = industry_concat[(industry_concat['PE1'] < limite_infierior_pe2) | (industry_concat['PE1'] > limite_superior_pe2)]

# Combinar outliers
outliers_combined = pd.concat([outliers, outliers_eg2, outliers_pe1, outliers_pe2], ignore_index=True)
outliers_combined = outliers_combined.drop_duplicates(subset='ticker', keep='first')

print()
print('ESTA ES LA LISTA DE OUTLIERS:')
print()
print(outliers_combined)

# Quitar outliers del dataset
industry_concat = industry_concat[~industry_concat['ticker'].isin(outliers_combined['ticker'])]

# Agrupar por industria
industry_groupby = industry_concat.groupby('industry').agg({
    'EG1': 'mean', 'EG2': 'mean', 'PE1': 'mean', 'PE2': 'mean', 'PEG1': 'mean', 'PEG2': 'mean'
})

industry_concat = industry_concat.iloc[:, [1] + list(range(5, 17)) + [df.columns.get_loc('market_cap_category')]]

print('ESTA ES LA LISTA ORIGINAL:')
print(industry_concat)

# Filtro: empresas por encima del promedio
def above_average(row):
    industry = row.get('industry', None)
    if industry and industry in industry_groupby.index:
        return (row['PE1'] > industry_groupby.loc[industry, 'PE1']) and \
               (row['PE2'] > industry_groupby.loc[industry, 'PE2'])
    return False

filtered_version = pd.concat([industry_concat[industry_concat.apply(above_average, axis=1)], industry_groupby])
filtered_version.fillna('', inplace=True)
filtered_version = filtered_version.loc[:, ~filtered_version.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
filtered_version = filtered_version.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]

print()
print('EMPRESAS POR ENCIMA DEL PROMEDIO')
print(filtered_version)
# Gráfico empresas por encima del promedio
ax = filtered_version.plot(kind='bar', x='ticker', rot=90, grid=True, title=f"{industry_look} (Above Average)", figsize=(12, 6))
eg1_avg = industry_groupby['EG1'].values[0]
eg2_avg = industry_groupby['EG2'].values[0]
ax.axhline(y=eg1_avg, color='blue', linestyle='--', linewidth=2)
ax.axhline(y=eg2_avg, color='orange', linestyle='--', linewidth=2)
ax.legend()
plt.tight_layout()
plt.show()

# Filtro: empresas por debajo del promedio
def below_average(row):
    industry = row.get('industry', None)
    if industry and industry in industry_groupby.index:
        return (row['PE1'] < industry_groupby.loc[industry, 'PE1']) and \
               (row['PE2'] < industry_groupby.loc[industry, 'PE2'])
    return False

filtered_below = industry_concat[industry_concat.apply(below_average, axis=1)]
filtered_below = filtered_below.loc[:, ~filtered_below.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
filtered_below = filtered_below.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]

print()
print('EMPRESAS POR DEBAJO DEL PROMEDIO')
print(filtered_below)

# Gráfico empresas por debajo del promedio
ax = filtered_below.plot(kind='bar', x='ticker', rot=90, grid=True, title=f"{industry_look} (Below Average)", figsize=(12, 6))
ax.axhline(y=eg1_avg, color='blue', linestyle='--', linewidth=2)
ax.axhline(y=eg2_avg, color='orange', linestyle='--', linewidth=2)
ax.legend()
plt.tight_layout()
plt.show()
