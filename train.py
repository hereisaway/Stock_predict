import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 超参数
LOOKBACK_WINDOW = 60
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后的输出
        return out

# 数据获取与处理
def get_data(stock_ticker):
    df = yf.download(stock_ticker, start='2010-01-01', end='2023-01-01')
    return df['Close'].values

def create_dataset(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# 主程序
if __name__ == "__main__":

    # 获取数据
    stock_ticker = 'AAPL'
    data = get_data(stock_ticker)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    # 创建数据集
    X, y = create_dataset(data, LOOKBACK_WINDOW)

    # 划分训练集和测试集
    split = int(len(X) * 0.8)  # 80% 训练集，20% 测试集
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 转换为张量
    X_train = torch.FloatTensor(X_train).view(-1, LOOKBACK_WINDOW, 1)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test).view(-1, LOOKBACK_WINDOW, 1)
    y_test = torch.FloatTensor(y_test)

    # 初始化模型、损失函数和优化器
    model = LSTM()
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

    # 预测
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train)
        test_predict = model(X_test)

    # 反归一化
    train_predict = scaler.inverse_transform(train_predict.numpy())
    test_predict = scaler.inverse_transform(test_predict.numpy())
    original_data = scaler.inverse_transform(data)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(np.arange(LOOKBACK_WINDOW, LOOKBACK_WINDOW + len(train_predict)), train_predict, label='Train Predict', color='red')
    plt.plot(np.arange(LOOKBACK_WINDOW + len(train_predict), LOOKBACK_WINDOW + len(train_predict) + len(test_predict)), test_predict, label='Test Predict', color='green')
    plt.legend()
    plt.title(f'Stock Price Prediction for {stock_ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()