# backtrader回测多个策略
## 简介
本仓库包含多个基于backtrader框架的交易策略，旨在提供多样化的市场交易方法。每个策略都有详细的实现代码和相应的原理解释，以便于学习和研究。

## 依赖环境
- Python 3.x
- backtrader
- numpy
- pandas
- scipy
matplotlib (用于绘图)
## 安装指南
确保Python已安装在你的系统上。
使用pip安装所需的库：
```
pip install backtrader numpy pandas scipy matplotlib
```
## 策略列表
### 配对交易(pair-trading)
- 原理：
基于两个具有高度相关性的股票价格之间的价差。当价格偏离到一定程度时，买入被低估的股票并卖出被高估的股票，期望它们最终回归到长期均衡状态。
