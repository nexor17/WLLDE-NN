import pandas as pd
import mplfinance as mpf


data = pd.read_csv('data/GSPC.csv')
data.index = pd.date_range(start='2015-01-01', periods=len(data))
data.rename(columns={'DayOfYear': 'Date'}, inplace=True)
mpf.plot(data, type='candle', style='charles', ylabel='Price')

