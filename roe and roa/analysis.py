import pandas as pd 
import requests 
from bs4 import BeautifulSoup



api_key = 'c8075ee98727510ad0e9ef13cc8cc5ef'
base_url = 'https://finviz.com/screener.ashx?v=111&f=earningsdate_nextweek'


data = []

for page in range(1,20):

	if page == 1:
		url = base_url
	else:
		start_record = 1+20*(page-1)
		url = f"{base_url}&r={start_record}"

	payload = {'api_key': api_key, 'url': url}

	print(f"Scrapeando pagina {page} en {url}")

	r = requests.get('https://api.scraperapi.com/', params=payload)

	soup = BeautifulSoup(r.text, 'lxml')

	tr_div = soup.find_all('tr', class_='styled-row is-bordered is-rounded is-hoverable is-striped has-color-text')



	for row in tr_div:
		a_div = row.find('a', class_='tab-link')
		a = a_div.get_text(strip=True) if a_div else 'na'
		data.append({'Ticker': a})

df = pd.DataFrame(data)


challenge = pd.read_csv('tradable_securities.csv')

newset = df.merge(challenge, on='Ticker')

newset.columns = [col.strip().lower().replace(' ', '_') for col in newset.columns]


newset.drop(['mic', 'company_name', 'market_capital_size_(in_millions)'], axis=1, inplace=True)

zack = pd.read_csv('1.csv')
zack.columns = [col.strip().lower().replace(' ', '_') for col in zack.columns]

newset = newset.merge(zack, on='ticker')

industries = newset['industry'].unique().tolist()

stocks_list = newset['ticker'].unique().tolist()

print(industries)
print('----------------------------------------------------------------------------------')
print(f"PARA ESTA SEMANA TENEMOS LA CANTIDAD DE {len(industries)} INDUSTRIAS QUE ANALIZAR")
print('----------------------------------------------------------------------------------')
print(stocks_list)
print('-----------------------------------------------------------------------------------')
print(f"ESTA SEMANA TENEMOS LA CANTIDAD DE {len(stocks_list)} ACCIONES QUE ANALIZAR")
print('-----------------------------------------------------------------------------------')