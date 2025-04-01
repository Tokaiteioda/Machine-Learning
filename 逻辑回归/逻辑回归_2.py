import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 防止报错
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import expit

data = np.loadtxt('ex2data2(1).txt',delimiter=',')
X = data[:,:2]   # 选取前两列(选到第3列但不包括第3列)
Y = data[:,2].reshape(-1,1)    # 只选取第3列

m = len(Y)       # 数据个数
poly = PolynomialFeatures(degree=6)  # 生成6阶多项式扩展
X_poly = poly.fit_transform(X)       # 自带偏置项


theta = np.zeros((X_poly.shape[1],1))      # 初始设定
learning_rate = 0.18
iterations = 10000
# 目标函数h(x) = Θ₁x₁ + Θ₂x₂ + Θ₃ = Θ.T * (x₁,x₂,1)  (Θ为参数矩阵)
# 逻辑函数g(x) = 1 / (1 + exp(-h(x)))
# 代价函数J(Θ,y) = (1 / m)(-y * log(g(x)) - (1-y) * log(1-g(x)))  (m为样本个数)
# 代价函数求导▽J(Θ) = (1/m) * x.T * (g(x)-y)   (x.T为样本矩阵)
# 参数更新Θ = Θ - α▽J(Θ)
def gradient_descent(x,y,theta,learning_rate,iterations):
    m = len(x)
    cost_history = []

    for i in range(iterations):
        predictions = 1 / (1 + np.exp(-x.dot(theta)))
        errors = predictions - y
        epsilon = 1e-8
        gradient = (1 / m) * x.T.dot(errors)
        theta = theta - learning_rate * gradient
        cost = (1 / m) * np.sum(-y * np.log(predictions + epsilon) - (1 - y) * np.log(1 + epsilon - predictions))
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"第{i}次迭代，代价函数为{cost}")
    return theta,cost_history


theta,cost_history = gradient_descent(X_poly,Y,theta,learning_rate,iterations)


data_np = np.array(data).reshape(-1,3)    # 绘图用
x = data_np[:,0]
y = data_np[:,1]

labels = data_np[:,2]                    # 布尔索引
a_mark = labels == 0
b_mark = labels == 1

x_min,x_max = x.min()-1,x.max()+1            # 画决策边界用
y_min,y_max = y.min()-1,y.max()+1
xx ,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))

grid_points = np.c_[xx.ravel(),yy.ravel()]
grid_points_poly = poly.transform(grid_points)

Z = expit(grid_points_poly.dot(theta))   # 决策边界计算(使用scipy库实现更为稳定的计算，防止溢出)
Z = Z.reshape(xx.shape)

fig ,ax = plt.subplots(1,2,figsize=(10,6))

ax[0].contour(xx,yy,Z,levels=[0.5],linewidths=2)    # 决策边界0.5为阈值
ax[0].scatter(x[a_mark],y[a_mark],marker='o',color='g')
ax[0].scatter(x[b_mark],y[b_mark],marker='x',color='r')

ax[1].scatter(range(iterations),cost_history,s=3)
plt.show()