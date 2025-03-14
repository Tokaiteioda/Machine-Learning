import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt

x_data = []
y_data = []
with open(r"ex1data1(1).txt") as file:  # 读取数据用于绘图
    for line in file:  # 逐行读取
        n = list(map(float, line.strip().split(",")))  # 去掉首尾空白，处理为float列表
        x_data.append(n[0])  # x列表
        y_data.append(n[1])  # y列表

data = [[a, b] for a, b in zip(x_data, y_data)]  # 转为numpy数组
data = np.array(data)
X = data[:, 0]  # 面积（x）
Y = data[:, 1]  # 价格（y）


                                                               # h(x) = Θ₁x + Θ₂ = Θ.T * (x , 1)  (Θ为参数矩阵)
                                                               # J(Θ₁,Θ₂) = (1/2m) * Σ (h(x)-y)²  (m为样本个数)
                                                               # ▽J(Θ) = (1/m) * x.T * (h(x)-y)   (x.T为样本矩阵)
                                                               # Θ = Θ - α▽J(Θ)
def gradient_descent(x, y, theta, learning_rate, iterations):  # 线性回归问题的代价函数是均方误差（MSE）
    m = len(y)         # 样本数量
    cost_history = []  # 迭代记录

    for i in range(iterations):                                # 梯度下降
        predictions = x.dot(theta)                             # 预测值计算（dot为矩阵乘法） hΘ = xΘ₁ + Θ₂
        errors = predictions - y                               # 误差计算

        gradient = (1 / m) * x.T.dot(errors)                   # 梯度计算
        theta = theta - learning_rate * gradient               # 梯度下降

        cost = (1 / (m * 2)) * np.sum(errors ** 2)             # 代价函数计算
        cost_history.append(cost)

        if i % 100 == 0 or i == 1000:
            print(f'第{i + 1}次梯度下降，代价函数：{cost}')

    return theta,cost_history

m = len(x_data)
X = np.c_[np.ones(m), X]                                       # 添加偏置列向量1

theta = np.zeros(2)                                            # 参数矩阵，学习率，迭代次数
learning_rate = 0.02
iterations = 1000

theta, cost_list = gradient_descent(X,Y,theta,learning_rate,iterations)
print(f"最终的参数为{theta}")

x_values = range(1,26)
y_values = [theta[1] * x + theta[0] for x in x_values]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].scatter(x_data, y_data, s=10)
ax[0].plot(x_values, y_values, linewidth=1,color="red")
ax[0].set_title("MarkerSize", fontsize=24)
ax[0].set_xlabel("Population of City in 10,000s", fontsize=14)
ax[0].set_ylabel("Profit in $10,000s", fontsize=14)

ax[1].plot(range(iterations), cost_list)
ax[1].set_xlabel('Iterations')
ax[1].set_ylabel('Cost (J)')
ax[1].set_title('Cost Function Convergence')
plt.show()

