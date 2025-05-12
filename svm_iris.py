import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
iris = datasets.load_iris()
# 只选择Setosa和Versicolor类
X = iris.data[iris.target != 2, :3]  # 只选择前三个特征
y = iris.target[iris.target != 2]  # 只选择前两类

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 训练SVM模型
model = svm.SVC(kernel='linear', C=1)
model.fit(X_scaled, y)

# 4. 绘制决策平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建网格以绘制决策边界
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
z_min, z_max = X_scaled[:, 2].min() - 1, X_scaled[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50),
                         np.linspace(z_min, z_max, 50))

# 预测每个点的类别
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

# 绘制决策边界
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, s=50, cmap='viridis')
ax.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], alpha=0.3, levels=np.linspace(-0.5, 1.5, 3), cmap='coolwarm')

# 设置标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('SVM Decision Boundary on Iris Dataset (Setosa vs Versicolor)')
plt.show()