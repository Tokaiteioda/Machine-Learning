import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt

x_data = []
y_data = []
z_data = []
data = []
with open("ex2data1(1).txt") as file:  # 读取数据
    for line in file:
        n = list(map(float, line.strip().split(',')))
        data.append(n)
        x_data.append(n[0])
        y_data.append(n[1])
        z_data.append(n[2])


def standardize(data):  # 标准化
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_std = (data - mean) / std
    return data_std


m = len(x_data)  # 数据个数
X = np.array(x_data).reshape(m, 1)  # 处理为numpy数组以及标准化
Y = np.array(y_data).reshape(m, 1)
Z = np.array(z_data).reshape(m, 1)

X = standardize(X)
Y = standardize(Y)
data_raw = np.array(data).reshape(m,3) # 绘图用
# 目标函数h(x) = Θ₁x₁ + Θ₂x₂ + Θ₃ = Θ.T * (x₁,x₂,1)  (Θ为参数矩阵)
# 逻辑函数g(x) = 1 / (1 + exp(-h(x)))
# 代价函数J(Θ,y) = (1 / m)(-y * log(g(x)) - (1-y) * log(1-g(x)))  (m为样本个数)
# 代价函数求导▽J(Θ) = (1/m) * x.T * (g(x)-y)   (x.T为样本矩阵)
# 参数更新Θ = Θ - α▽J(Θ)
def gradient_descent(x, y, theta, learning_rate, iterations):  # 梯度下降
    m = len(x)  # 数据个数
    cost_history = [] # 代价记录

    for i in range(iterations):
        predictions = 1 / (1 + np.exp(-x.dot(theta)))      # 预测值计算（dot为矩阵乘法）
                                                           # g(x) = 1 / (1 + exp(-h(x)))
        errors = predictions - y                           # 误差计算
        epsilon = 1e-8  # 避免 log(0)

        gradient = (1 / m) * x.T.dot(errors)               # 梯度计算
        theta = theta - learning_rate * gradient           # 梯度下降

        cost = (1 / m) * np.sum(-1 * y * np.log(predictions + epsilon) - (1 - y) * np.log(1 - predictions + epsilon))  # 代价函数J计算
        cost_history.append(cost)

        if i % 100 == 0 or i == 9999:
            print(f"第{i}次梯度下降，当前代价函数为{cost}")

    return theta,cost_history

DATA = np.c_[X,Y,np.ones(m)]   # 添加偏置项1
theta = np.zeros((3,1)) # 设置起点,学习率,迭代次数
learning_rate = 0.02
iterations = 10000

theta,cost_list = gradient_descent(DATA,Z,theta,learning_rate,iterations)  # 执行
print(f"结果为{theta[0]}x₁+{theta[1]}x₂+{theta[2]}")

x = data_raw[:,0]                # 绘图数据处理
y = data_raw[:,1]

labels = data_raw[:,2]           # 布尔索引
pass_mask = labels == 1      # 合格
fail_mask = labels == 0      # 不合格

theta_1 = theta[0,0]
theta_2 = theta[1,0]
theta_3 = theta[2,0]

# 计算决策边界
x_min, x_max = x.min(), x.max()  # 获取原始数据的范围
x_boundary = np.linspace(x_min, x_max, 100)  # 原始 x1 范围

# 对 x_boundary 进行标准化,用于计算标准化y_boundary
x_boundary_std = (x_boundary - np.mean(x_data)) / np.std(x_data)

# 使用标准化数据计算 y_boundary
y_boundary_std = - (theta_1 * x_boundary_std + theta_3) / theta_2

# 反标准化 y_boundary，使其回到原始数据尺度
y_boundary = y_boundary_std * np.std(y_data) + np.mean(y_data)

fig, ax = plt.subplots(1,2,figsize=(10,6))

# 画决策边界
ax[0].plot(x_boundary, y_boundary, 'b-', label="Decision Boundary")
ax[0].legend()

ax[0].scatter(x[pass_mask],y[pass_mask],marker="o",color="g",label='pass')
ax[0].scatter(x[fail_mask],y[fail_mask],marker="x",color="r",label='fail')


ax[1].plot(range(iterations),cost_list)

plt.show()
