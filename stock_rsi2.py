import os
import datetime
import numpy as np
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import talib

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
MAX_TIMEPERIOD = 40
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
        df_stock.to_pickle(fname)
    return df_stock


def calc_change(df, dates, days=30):
    changes = []
    for date in dates:
        d = df.loc[date:date + datetime.timedelta(days=days)]
        change = list((d['Adj Close'] / d.iloc[0]['Adj Close']))
        changes.append(change)
    return changes


def main():
    df_stock = get_stock(TICKER, START_DATE, END_DATE)

    df_upper = pd.DataFrame()
    df_lower = pd.DataFrame()

    for timeperiod in np.arange(5, MAX_TIMEPERIOD + 1):
        df = df_stock.copy()
        df['RSI'] = talib.RSI(df['Adj Close'].values, timeperiod=timeperiod)
        df = df.dropna()

        # RSI上限
        results = []
        for upper_limit in np.arange(60, 90):
            # RSIが上限を超えた日付を探す
            dates = df[(df['RSI'].shift() < upper_limit) &
                       (df['RSI'] >= upper_limit)].index
            # RSIが上限を超えた日を基準に株価の変化率を求める
            changes = calc_change(df, dates)
            # 結果に追加
            results.append(
                {'limit': upper_limit, 'average': np.average(sum(changes, []))})

        # 結果をDataFrameに変換
        d = pd.DataFrame(results)
        d['timeperiod'] = timeperiod
        # 変化率を%にし基準を0とする
        d['average'] = d['average'] * 100 - 100
        df_upper = pd.concat([df_upper, d])

        # RSI下限
        results = []
        for lower_limit in np.arange(10, 40):
            # RSIが下限を下回った日付を探す
            dates = df[(df['RSI'].shift() > lower_limit) &
                       (df['RSI'] <= lower_limit)].index
            # RSIが下限を下回った日を基準に株価の変化率を求める
            changes = calc_change(df, dates)
            # 結果に追加
            results.append(
                {'limit': lower_limit, 'average': np.average(sum(changes, []))})

        # 結果をDataFrameに変換
        d = pd.DataFrame(results)
        d['timeperiod'] = timeperiod
        # 変化率を%にし基準を0とする
        d['average'] = d['average'] * 100 - 100
        df_lower = pd.concat([df_lower, d])

    # グラフ化
    fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
    for timeperiod in df_upper['timeperiod'].unique():
        d = df_upper[df_upper['timeperiod'] == timeperiod]
        axes[0].plot(d['limit'], d['average'], label=timeperiod)
    for timeperiod in df_lower['timeperiod'].unique():
        d = df_lower[df_lower['timeperiod'] == timeperiod]
        axes[1].plot(d['limit'], d['average'], label=timeperiod)
    for i in range(2):
        axes[i].axhline(y=0, color='red', linestyle=':')
        axes[i].grid()
        axes[i].set_xlabel('RSI閾値')
        axes[i].set_ylabel('変化率(%)')
    fig.suptitle('RSI閾値と変化率')
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3)
    axes[0].set_title('RSI上限値を超えた後')
    axes[1].set_title('RSI下限値を下回った後')
    plt.savefig('stock_rsi2_1.png')
    plt.show()
    plt.close()

    # ヒートマップ
    fig, axes = plt.subplots(nrows=2, figsize=(8, 16))
    sns.heatmap(df_upper.pivot('timeperiod', 'limit', 'average'),
                square=True, cmap='Blues_r', ax=axes[0])
    sns.heatmap(df_lower.pivot('timeperiod', 'limit', 'average'),
                square=True, cmap='Reds', ax=axes[1])
    for i in range(2):
        axes[i].invert_yaxis()
        axes[i].set_xlabel('RSI閾値')
        axes[i].set_ylabel('RSI期間(日)')
    axes[0].set_title('RSI上限閾値と株価変化率')
    axes[1].set_title('RSI下限閾値と株価変化率')
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3)
    plt.savefig('stock_rsi2_2.png')
    plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
    for timeperiod in df_upper.sort_values('average')['timeperiod'].unique()[0:5]:
        d = df_upper[df_upper['timeperiod'] == timeperiod]
        axes[0].plot(d['limit'], d['average'], label=timeperiod)
    for timeperiod in df_lower.sort_values('average', ascending=False)['timeperiod'].unique()[0:5]:
        d = df_lower[df_lower['timeperiod'] == timeperiod]
        axes[1].plot(d['limit'], d['average'], label=timeperiod)
    for i in range(2):
        axes[i].axhline(y=0, color='red', linestyle=':')
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel('RSI閾値')
        axes[i].set_ylabel('変化率(%)')
    fig.suptitle('RSI閾値と変化率')
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3)
    axes[0].set_title('RSI上限値を超えた後')
    axes[1].set_title('RSI下限値を下回った後')
    # plt.savefig('stock_rsi2_3.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
