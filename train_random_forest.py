import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 获取苹果（AAPL）的历史数据
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# 创建特征和目标变量
data['Lag_1'] = data['Close'].shift(1)  # 前一天的收盘价
data.dropna(inplace=True)  # 删除缺失值

# 特征和目标
X = data[['Lag_1']]
y = data['Close']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(X_train.max())

# 构建随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices', color='blue', marker='o')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Prices', color='orange', marker='x')
plt.title('AAPL Stock Price Prediction using Random Forest')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()