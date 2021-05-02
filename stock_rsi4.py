import os
import datetime
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
MAX_TIMEPERIOD = 40
TICKER = '^N225'
INIT_CASH = 1000000


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


class RsiStrategy(Strategy):
    '''
    RSI Strategy
    '''
    timeperiod = 14
    rsi_upper = 80
    rsi_lower = 20

    def init(self):
        close = self.data['Adj Close']
        self.rsi = self.I(talib.RSI, close, self.timeperiod)

    def next(self):
        '''
        RSIが上限値を超えたら買われすぎと判断し売る
        RSIが下限値を下回ったら売られすぎと判断し買う
        '''
        if crossover(self.rsi_lower, self.rsi):
            self.buy()
        elif crossover(self.rsi, self.rsi_upper):
            self.sell()


def main():
    df = get_stock(TICKER, START_DATE, END_DATE)

    bt = Backtest(
        df,
        RsiStrategy,
        cash=INIT_CASH,
        trade_on_close=False,
        exclusive_orders=True
    )

    stats, heatmap = bt.optimize(
        timeperiod=range(5, 41),
        rsi_upper=range(60, 91),
        rsi_lower=range(10, 41),
        return_heatmap=True)

    print(stats)
    bt.plot()
    plot_heatmaps(heatmap, agg='mean', plot_width=2048, filename='heatmap')

    # 3Dヒートマップで表示
    d = heatmap.reset_index()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(d['timeperiod'], d['rsi_upper'], d['rsi_lower'], c=d['SQN'])
    ax.set_xlabel('RSI期間(日)')
    ax.set_ylabel('RSI上限')
    ax.set_zlabel('RSI下限')
    plt.show()


if __name__ == '__main__':
    main()
