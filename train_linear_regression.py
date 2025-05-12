import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 获取苹果（AAPL）的历史数据
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# 创建特征和目标变量
data['Lag_1'] = data['Close'].shift(1)  # 前一天的收盘价
data.dropna(inplace=True)  # 删除缺失值

# 特征和目标
X = data[['Lag_1']]
y = data['Close']

# 对目标变量进行对数变换
y_log = np.log(y)

# 拆分数据集
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42, shuffle=False)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train_log)

# 进行预测
y_pred_log = model.predict(X_test)

# 逆变换
y_pred = np.exp(y_pred_log)

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(np.exp(y_test_log), y_pred))
print(f'RMSE: {rmse:.2f}')

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test_log):], np.exp(y_test_log), label='Actual Prices', color='blue', marker='o')
plt.plot(data.index[-len(y_test_log):], y_pred, label='Predicted Prices', color='orange', marker='x')
plt.title('AAPL Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()