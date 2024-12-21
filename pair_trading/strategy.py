import warnings
import numpy as np
import pandas as pd
import backtrader as bt
from scipy.stats import t
# from bigdatasource import DataSource


# 线性回归
def linear(x, y, sigma:float=1):
    """
    线性回归
    """
    x = x.reshape(-1, 1)
    I = np.ones_like(x)
    y = y.reshape(-1, 1)
    X = np.concatenate([I, x], axis=-1)
    cov = np.linalg.pinv(X.T@X) * sigma
    beta = cov@X.T@y

    # 截距是否显著(来检验二者是否存在套利空间)
    t_value = beta[0][0] / np.sqrt(cov[0][0])
    ddof = len(x) - 2
    critical_t_value = t.ppf(1 - 0.05 / 2, ddof)
    is_significant = abs(t_value) > critical_t_value

    # 计算价差
    resid = y - x * beta[1][0]

    return resid, beta[1][0]

# 数据预处理
def data_processing(data):
    """数据日期对齐!!!!!!!!!!!!!!!!!!!!"""
    columns = list(data.columns)
    columns.remove('date')
    columns.remove('instrument')
    df = pd.pivot(data, index='date', columns='instrument', values=columns).ffill().dropna()
    df = df.reset_index()
    df.columns = ['date'] + ['__'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns[1:]]
    df_long = pd.melt(df, id_vars='date', var_name='variable', value_name='value')
    df_long[['type', 'instrument']] = df_long['variable'].str.split('__', expand=True)
    df_long = df_long.drop(columns='variable')
    df_long = df_long[['date', 'instrument', 'type', 'value']]
    df_long = df_long.pivot_table(index=['date', 'instrument'], columns='type', values='value', aggfunc='first').reset_index()
    df_long.columns.name = None
    return df_long

class MyCerebro:
    def __init__(self, data, initial_cash: float=10000, commission: float=0.0002):
        """
        :params data: 传入数据, 包括date, instrument, 其他因子列
        :params initial_cash: 初始资金
        :params commission: 佣金
        """
        # 参数
        self.initial_cash = initial_cash
        self.commission = commission

        # 数据预处理
        self.data = data
        self.data['date'] = pd.to_datetime(data['date'])
        self.data.sort_values(['date', 'instrument'], inplace=True)

        # 创建大脑
        self.cerebro = self._make_cerebro()
    
    def _make_cerebro(self):
        """
        产生一个新的Cerebro
        """
        warnings.filterwarnings('ignore')
        data = self.data.copy()
        instruments = list(data.groupby('instrument').apply(len).sort_values(ascending=False).to_dict().keys())
        columne = list(data.columns)
        columne.remove('date')
        columne.remove('instrument')

        # 创建数据流
        class Mydata(bt.feeds.PandasData):
            lines = tuple(columne)
            params = tuple([(i, i) for i in columne])

        # 创建大脑
        cerebro = bt.Cerebro()

        # 将数据喂给cerebro
        for ins in instruments:
            df = Mydata(
                dataname=data[data['instrument']==ins], 
                name=ins, 
                datetime='date'
            )
            cerebro.adddata(df, name=ins)
        
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.broker.setcash(self.initial_cash)
        
        # print('数据加载完成')
        return cerebro

class Pair_trading(bt.Strategy):
    params = (
        ('lookback', 50),
    )
    def __init__(self):
        self.instruments = [p._name for p in self.datas]
        self.values = []
        self.date = []
        
        self.ins_1 = []
        self.ins_2 = []
        
        self.cant_trade = []

        self.history_score = 0

    def notify_order(self, order):
        """
        订单执行和记录
        """
        dt = self.data.datetime.date(0).strftime('%Y-%m-%d')
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'{dt} 时刻买入 {order.data._name} 成功, 执行价格 {order.executed.price}, 买入数量 {order.executed.size}')

            elif order.issell():
                print(f'{dt} 时刻卖出 {order.data._name} 成功, 执行价格 {order.executed.price}, 卖出数量 {order.executed.size}')

        
        if order.status in [order.Canceled]:
            print(f'{dt} 时刻的订单被取消')

        if order.status in [order.Margin]:
            print(f'{dt} 时刻的保证金不足, 交易无法继续')

        if order.status in [order.Rejected]:
            print(f'{dt} 时刻的订单委托被拒绝')
    
    def order_percent(self, instrument, ratio):
        return

    def next(self):
        # print('当前总资产:', self.broker.getvalue())
        self.cant_trade = []

        # 时间索引和k线索引(索引从1开始)
        dt = self.data.datetime.date(0).strftime('%Y-%m-%d')
        current_bar_index = len(self)
        self.date.append(dt)
        self.values.append(self.broker.getvalue())
        self.ins_1.append(self.getdatabyname(self.instruments[0]).open[0])
        self.ins_2.append(self.getdatabyname(self.instruments[1]).open[0])

        # ========================止盈止损模块==========================(这一块容易出现重复交易)
        for ins in self.instruments:
            if self.getpositionbyname(ins).size>0:
                cost_price = self.getpositionbyname(ins).price
                last_price = self.getdatabyname(ins).open[0]
                if last_price / cost_price - 1 > 0.02:
                    # print(dt, ins, '止盈')
                    self.order_target_percent(ins, 0)
                    self.cant_trade.append(ins)
                elif last_price / cost_price - 1 < -0.05:
                    # print(dt, ins, '止损')
                    self.order_target_percent(ins, 0)
                    self.cant_trade.append(ins)

        if current_bar_index <= self.params.lookback:
            return
        
        # 列表中的第一支票为x, 第二只票为y
        # x = np.array([self.getdatabyname(self.instruments[0]).open[p] for p in range(-current_bar_index+1, 1)])
        # y = np.array([self.getdatabyname(self.instruments[1]).open[p] for p in range(-current_bar_index+1, 1)])
        
        # 列表中的第一支票为x, 第二只票为y
        x = np.array([self.getdatabyname(self.instruments[0]).close[p] for p in range(-self.params.lookback+1, 1)])
        y = np.array([self.getdatabyname(self.instruments[1]).close[p] for p in range(-self.params.lookback+1, 1)])

        # 设置交易头寸
        resid, beta = linear(np.log(x), np.log(y))
        unit = 100
        if beta > 1:
            ins_0_unit = unit
            ins_1_unit = (unit * beta) - ((unit * beta) % unit)
        else:
            ins_1_unit = unit
            ins_0_unit = (unit / beta) - ((unit / beta) % unit)

        # 结果显著, 所以存在套利空间
        std = np.std(resid)
        avg = np.mean(resid)
        score = (resid - avg) / std
        resid = score[-1]
        if resid > 1 and self.history_score < 1:
            self.order_target_percent(self.instruments[1], 0)
            if self.getpositionbyname(self.instruments[0]).size == 0 and self.instruments[0] not in self.cant_trade:
                self.buy(self.instruments[0], ins_0_unit)

        
        elif resid < -1 and self.history_score > -1:
            self.order_target_percent(self.instruments[0], 0)
            if self.getpositionbyname(self.instruments[1]).size == 0 and self.instruments[1] not in self.cant_trade:
                # print(f'{dt} 买入 {self.instruments[1]} 卖出 {self.instruments[0]}')
                self.buy(self.instruments[1], ins_1_unit)
                # self.order_target_percent(self.instruments[1], 0.5)

    def stop(self):
        data = pd.DataFrame(
            {
                'date': self.date, 
                'value': self.values, 
                'ins_1': self.ins_1, 
                'ins_2': self.ins_2
            }
        )
        data['value'] = data['value'] / data['value'].iloc[0] - 1
        data['ins_1'] = data['ins_1'] / data['ins_1'].iloc[0] - 1
        data['ins_2'] = data['ins_2'] / data['ins_2'].iloc[0] - 1
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig = plt.figure()
        plt.plot(data['date'], data['value'], label='strategy')
        # plt.plot(data['date'], data['ins_1'], label='instrument1')
        # plt.plot(data['date'], data['ins_2'], label='instrument2')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    import pandas as pd
    # instruments = ['002231.SZ', '002579.SZ']
    data = pd.read_csv('data.csv')
    
    # data['close'] /= data['adjust_factor']
    # data['open'] /= data['adjust_factor']
    # data['high'] /= data['adjust_factor']
    # data['low'] /= data['adjust_factor']
    
    
    data = data_processing(data)

    # print(data)
    # data.to_csv('test.csv', index=False)
    # import pandas as pd

    # data = pd.read_csv('test.csv')

    cerebro = MyCerebro(data, 10000, 0.0003).cerebro
    cerebro.addstrategy(Pair_trading, lookback=20)
    cerebro.run()
    # cerebro.plot()

