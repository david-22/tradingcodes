import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Cargar datos
d = pd.read_csv('1.csv')
df = pd.DataFrame(data=d)

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df = df.loc[:, ~df.columns.str.startswith('unnamed')]

df.rename(columns={
    'last_close': 'price',
    'last_yr`s_eps_(f0)_before_nri': 'eps0',
    'f1_consensus_est.': 'eps1',
    'f2_consensus_est.': 'eps2',
    'market_cap_(mil)': 'marketcap'
}, inplace=True)

df = df[df['exchange'] != 'OTC']
df.dropna(subset=['eps0', 'eps1', 'eps2', 'company_name'], inplace=True)
df.reset_index(drop=True, inplace=True)

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

df['PE1'] = df['price'] / df['eps1']
df['PE2'] = df['price'] / df['eps2']
df['PEG1'] = (df['PE1'] / (df['EG1'] * 100)) * 100
df['PEG2'] = (df['PE2'] / (df['EG2'] * 100)) * 100

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

analysis = ['Transportation - Airline', 'Computer - Integrated Systems', 'REIT and Equity Trust - Other', 'Security and Safety Services', 'Financial - Investment Management', 'Insurance - Brokerage', 'Manufacturing - Electronics', 'Electronics - Connectors', 'Banks - Midwest', 'Computers - IT Services', 'Banks - Northeast', 'Containers - Paper and Packaging', 'Banks - Southwest', 'Financial - Miscellaneous Services', 'Financial - Savings and Loan', 'Oil and Gas - Field Services', 'Banks - Major Regional', 'Instruments - Control', 'Medical - Products', 'Banks - Southeast', 'Agriculture - Products', 'Internet - Software', 'Banks - West', 'Insurance - Property and Casualty', 'Building Products - Home Builders', 'Cable Television', 'Securities and Exchanges', 'Retail - Restaurants', 'Medical - HMOs', 'Utility - Electric Power', 'Oil and Gas - Exploration and Production - United States', 'Financial - Consumer Loans', 'Beverages - Soft drinks', 'Transportation - Rail', 'Medical - Outpatient and Home Healthcare', 'Medical Services', 'Manufacturing - General Industrial ', 'Chemical - Diversified ', 'REIT and Equity Trust', 'Consulting Services', 'Solar', 'REIT and Equity Trust - Retail', 'Medical - Instruments', 'Financial Transaction Services', 'Mining - Non Ferrous', 'Real Estate - Development', 'Aerospace - Defense', 'Insurance - Accident and Health', 'Automotive - Domestic', 'Internet - Services', 'Automotive - Retail and Wholesale - Parts', 'Insurance - Multi line', 'Toys - Games - Hobbies', 'Medical - Hospital', 'Business - Services', 'Hotels and Motels', 'Business - Office Products', 'Diversified Operations', 'Film and Television Production and Distribution', 'Semiconductor - General', 'Advertising and Marketing', 'Communication - Components', 'Automotive - Replacement Parts', 'Gaming', 'Food - Miscellaneous', 'Financial - Investment Bank', 'Textile - Home Furnishing', 'Chemical - Specialty', 'Semiconductor - Analog and Mixed', 'Mining - Gold', 'Financial - Mortgage & Related Services', 'Tobacco', 'Waste Removal Services', 'Leisure and Recreation Products', 'Oil and Gas - Refining and Marketing', 'Staffing Firms', 'Building Products - Maintenance Service', 'Automotive - Retail and Whole Sales', 'Publishing - Books', 'Wireless National', 'Aerospace - Defense Equipment', 'Electronics - Miscellaneous Components', 'Automotive - Original Equipment', 'Outsourcing', 'Retail - Miscellaneous', 'Building Products - Miscellaneous', 'Medical - Drugs', 'Medical - Biomedical and Genetics', 'REIT and Equity Trust - Residential', 'Internet - Software and Services', 'Transportation - Equipment and Leasing', 'Medical - Dental Supplies', 'Building Products - Wood']

highlight_tickers = ['AAL', 'AGYS', 'ALEX', 'ALK', 'ALLE', 'AMP', 'AON', 'AOS', 'APH', 'ARE', 'ASB', 'ASGN', 'AUB', 'AVY', 'BANC', 'BFH', 'BHLB', 'BKR', 'BKU', 'BMI', 'BRKL', 'BSX', 'CADE', 'CALM', 'CALX', 'CATY', 'CB', 'CBU', 'CCI', 'CCS', 'CHTR', 'CLB', 'CME', 'CMG', 'CNC', 'CNP', 'CNX', 'COF', 'COKE', 'COOP', 'CSGP', 'CSX', 'CUBI', 'DCOM', 'DGX', 'DHI', 'DHR', 'DLR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DX', 'EBC', 'EFX', 'EGBN', 'ENPH', 'ENVA', 'EPRT', 'EW', 'FBP', 'FCFS', 'FCX', 'FFBC', 'FFIC', 'FOR', 'FRME', 'GBCI', 'GD', 'GL', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GSHD', 'GTY', 'HAL', 'HAS', 'HCA', 'HCSG', 'HFWA', 'HLT', 'HNI', 'HON', 'IBM', 'IMAX', 'INTC', 'IPG', 'IQV', 'ISRG', 'ITGR', 'IVZ', 'JOE', 'KDP', 'KEY', 'KN', 'KO', 'LBRT', 'LH', 'LHX', 'LKQ', 'LMT', 'LOB', 'LUV', 'LVS', 'LW', 'MC', 'MCO', 'MHK', 'MHO', 'MOH', 'MSCI', 'MTDR', 'MTH', 'MTX', 'MXL', 'NBHC', 'NDAQ', 'NEE', 'NEM', 'NOC', 'NOW', 'NTRS', 'NTST', 'NXPI', 'OCFC', 'OII', 'ONB', 'ORLY', 'OTIS', 'PCAR', 'PFS', 'PFSI', 'PHM', 'PKG', 'PM', 'PMT', 'PNR', 'POOL', 'POR', 'PRG', 'PSX', 'RHI', 'RJF', 'RNST', 'ROL', 'RTX', 'SAH', 'SBCF', 'SCHL', 'SHW', 'SIGI', 'SKYW', 'STC', 'STEL', 'SXT', 'SYF', 'T', 'TBBK', 'TDY', 'TEL', 'THRM', 'TMHC', 'TMO', 'TMUS', 'TNET', 'TPH', 'TRMK', 'TSCO', 'TSLA', 'TXN', 'TXT', 'UNP', 'URI', 'USNA', 'UVE', 'VBTX', 'VC', 'VICR', 'VKTX', 'VLO', 'VRE', 'VRSN', 'VZ', 'WAB', 'WFRD', 'WRB', 'WSFS', 'WST', 'WY', 'ZION']

# Crear un ExcelWriter con xlsxwriter
with pd.ExcelWriter('industries_analysis.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book

    for industry_look in analysis:
        industry_specified = df[df['industry'] == industry_look]

        if industry_specified.empty:
            continue

        industry_groupby = industry_specified.groupby('industry').agg({
            'EG1': 'mean', 'EG2': 'mean', 'PE1': 'mean', 'PE2': 'mean', 'PEG1': 'mean', 'PEG2': 'mean'
        })

        # Definir filtros
        def above_average(row):
            return (row['PE1'] > industry_groupby.loc[row['industry'], 'PE1']) and \
                   (row['PE2'] > industry_groupby.loc[row['industry'], 'PE2'])

        def below_average(row):
            return (row['PE1'] < industry_groupby.loc[row['industry'], 'PE1']) and \
                   (row['PE2'] < industry_groupby.loc[row['industry'], 'PE2'])

        # Filtrado
        filtered_version = pd.concat([industry_specified[industry_specified.apply(above_average, axis=1)], industry_groupby])
        filtered_version.fillna('', inplace=True)
        filtered_version = filtered_version.loc[:, ~filtered_version.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
        filtered_version = filtered_version.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]

        filtered_version_below = pd.concat([industry_specified[industry_specified.apply(below_average, axis=1)], industry_groupby])
        filtered_version_below.fillna('', inplace=True)
        filtered_version_below = filtered_version_below.loc[:, ~filtered_version_below.columns.isin(['price', 'eps0', 'eps1', 'eps2'])]
        filtered_version_below = filtered_version_below.loc[:, ['ticker', 'EG1', 'EG2', 'PE1', 'PE2', 'PEG1', 'PEG2']]

        eg1_avg = industry_groupby['EG1'].values[0]
        eg2_avg = industry_groupby['EG2'].values[0]

        # Generar gráficos en memoria para "Above Average"
        colors = ['orange' if ticker in highlight_tickers else 'gray' for ticker in filtered_version['ticker']]
        fig, ax = plt.subplots(figsize=(12, 6))
        filtered_version.plot(kind='bar', x='ticker', rot=90, grid=True, ax=ax, color=colors, title=f"{industry_look} (Above Average)")
        for label in ax.get_xticklabels():
            if label.get_text() in highlight_tickers:
                label.set_fontweight('bold')
        ax.axhline(y=eg1_avg, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=eg2_avg, color='green', linestyle='--', linewidth=2)
        ax.legend()
        plt.tight_layout()

        imgdata_above = BytesIO()
        plt.savefig(imgdata_above, format='png')
        plt.close()
        imgdata_above.seek(0)

        # Gráfico para "Below Average"
        colors_below = ['orange' if ticker in highlight_tickers else 'gray' for ticker in filtered_version_below['ticker']]
        fig, ax = plt.subplots(figsize=(12, 6))
        filtered_version_below.plot(kind='bar', x='ticker', rot=90, grid=True, ax=ax, color=colors_below, title=f"{industry_look} (Below Average)")
        for label in ax.get_xticklabels():
            if label.get_text() in highlight_tickers:
                label.set_fontweight('bold')
        ax.axhline(y=eg1_avg, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=eg2_avg, color='green', linestyle='--', linewidth=2)
        ax.legend()
        plt.tight_layout()

        imgdata_below = BytesIO()
        plt.savefig(imgdata_below, format='png')
        plt.close()
        imgdata_below.seek(0)

        # Añadir hoja con nombre de la industria
        sheet_name = industry_look[:31]  # Excel limita el nombre a 31 caracteres
        worksheet = workbook.add_worksheet(sheet_name)

        # Insertar imágenes en hoja Excel
        worksheet.insert_image('B2', f'{sheet_name}_above.png', {'image_data': imgdata_above, 'x_scale': 0.8, 'y_scale': 0.8})
        worksheet.insert_image('B40', f'{sheet_name}_below.png', {'image_data': imgdata_below, 'x_scale': 0.8, 'y_scale': 0.8})

        # Opcional: escribir tablas de datos en Excel (puedes comentarlo si solo quieres imágenes)
        filtered_version.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
        filtered_version_below.to_excel(writer, sheet_name=sheet_name, startrow=38, startcol=0, index=False)

    # El Excel se guarda automáticamente al salir del bloque 'with'
