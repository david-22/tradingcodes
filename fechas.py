import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

#df.to_excel('PE data filtered.xlsx', sheet_name = 'TABLE',index=False)


### Análisis por industria
industry_look = 'Furniture'
industry_specified = df[df['industry'] == industry_look]

industry_groupby = industry_specified.groupby('industry').agg({
    'EG1': 'mean', 'EG2': 'mean', 'PE1': 'mean', 'PE2': 'mean', 'PEG1': 'mean', 'PEG2': 'mean'
})
industry_concat = pd.concat([industry_specified, industry_groupby])
industry_concat.fillna('', inplace=True)

print(industry_concat.iloc[:, [1] + list(range(5, 16)) + [df.columns.get_loc('market_cap_category')]])

# Filtros para gráficos
def above_average(row):
    return (row['PE1'] > industry_groupby.loc[row['industry'], 'PE1']) and \
           (row['PE2'] > industry_groupby.loc[row['industry'], 'PE2'])

def below_average(row):
    return (row['PE1'] < industry_groupby.loc[row['industry'], 'PE1']) and \
           (row['PE2'] < industry_groupby.loc[row['industry'], 'PE2'])

# Empresas por encima del promedio
filtered_version = pd.concat([industry_specified[industry_specified.apply(above_average, axis=1)], industry_groupby])
filtered_version.fillna('', inplace=True)
filtered_version = filtered_version.loc[:, ~filtered_version.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
filtered_version = filtered_version.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]
print(filtered_version)

# Gráfico de empresas por encima del promedio
ax = filtered_version.plot(kind='bar', x='ticker', rot=90, grid=True, title=f"{industry_look} (Above Average)", figsize=(12, 6))

# Añadir líneas de promedio EG1 y EG2
eg1_avg = industry_groupby['EG1'].values[0]
eg2_avg = industry_groupby['EG2'].values[0]

ax.axhline(y=eg1_avg, color='red', linestyle='--', linewidth=2)
ax.axhline(y=eg2_avg, color='green', linestyle='--', linewidth=2)
ax.legend()
plt.tight_layout()
plt.show()

# Empresas por debajo del promedio
filtered_version_below = pd.concat([industry_specified[industry_specified.apply(below_average, axis=1)], industry_groupby])
filtered_version_below.fillna('', inplace=True)
filtered_version_below = filtered_version_below.loc[:, ~filtered_version_below.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
filtered_version_below = filtered_version_below.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]
print(filtered_version_below)

# Gráfico de empresas por debajo del promedio
ax = filtered_version_below.plot(kind='bar', x='ticker', rot=90, grid=True, title=f"{industry_look} (Below Average)", figsize=(12, 6))

# Añadir líneas de promedio EG1 y EG2
ax.axhline(y=eg1_avg, color='red', linestyle='--', linewidth=2)
ax.axhline(y=eg2_avg, color='green', linestyle='--', linewidth=2)
ax.legend()
plt.tight_layout()
plt.show()
