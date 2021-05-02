import os
import datetime
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
import talib

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
TICKER = '^N225'


def get_stock(ticker, start_date, end_date):
    '''
    get stock data from Yahoo Finance
    '''
    dirname = '../data'
    os.makedirs(dirname, exist_ok=True)
    period = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    fname = f'{dirname}/{ticker}_{period}.pkl'
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
    else:
        df_stock = pandas_datareader.data.DataReader(
            ticker, 'yahoo', start_date, end_date)
    return df_stock


def main():
    df = get_stock(TICKER, START_DATE, END_DATE)

    df['RSI'] = talib.RSI(df['Adj Close'].values, timeperiod=14)
    df = df.dropna()

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    axes[0].set_title('RSI')
    axes[0].plot(df.index, df['Adj Close'], label=f'{TICKER} Close')
    axes[0].grid()
    axes[0].legend(loc='upper left')
    axes[1].plot(df['RSI'], label='RSI 14days')
    axes[1].axhline(y=20, color='blue', linestyle=':')
    axes[1].axhline(y=30, color='red', linestyle=':')
    axes[1].axhline(y=70, color='red', linestyle=':')
    axes[1].axhline(y=80, color='blue', linestyle=':')
    axes[1].grid()
    axes[1].legend(loc='upper left')
    plt.savefig('stock_rsi1.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
