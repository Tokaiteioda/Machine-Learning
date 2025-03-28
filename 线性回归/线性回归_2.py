import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt


x_data = []
y_data = []
z_data = []
with open(r"ex1data2(1).txt") as file:  # 读取数据用于绘图
    for line in file:  # 逐行读取
        n = list(map(float, line.strip().split(",")))  # 去掉首尾空白，处理为float列表
        x_data.append(n[0])  # x列表
        y_data.append(n[1])  # y列表
        z_data.append(n[2])  # z列表


def standardize(data):
    """
    对传入的 numpy 数组进行 Z-score 标准化。

    参数:
        data (np.ndarray): 一维或二维数组，按列标准化。

    返回:
        data_std (np.ndarray): 标准化后的数据
        mean (np.ndarray): 每列均值
        std (np.ndarray): 每列标准差
    """
    mean = np.mean(data, axis=0)  # 均值
    std = np.std(data, axis=0)  # 标准差

    data_std = (data - mean) / std  # x(标准化) = (x - μ) / σ
    return data_std


X = np.array(x_data)                    # 转为numpy数组及标准化
X = standardize(X)
Y = np.array(y_data)
Y = standardize(Y)
Z = np.array(z_data)
Z = standardize(Z)


# 目标函数 h(x) = Θ₁x + Θ₂ = Θ.T * (x , 1)  (Θ为参数矩阵)
# 代价函数J(Θ,y) = (1/2m) * Σ (h(x)-y)²  (m为样本个数)
# 代价函数求导▽J(Θ) = (1/m) * x.T * (h(x)-y)   (x.T为样本矩阵)
# 参数更新Θ = Θ - α▽J(Θ)
def gradient_descent(x, y, theta, learning_rate, iterations):  # 线性回归问题的代价函数是均方误差（MSE）
    m = len(y)  # 样本数量
    cost_history = []  # 迭代记录

    for i in range(iterations):  # 梯度下降
        predictions = x.dot(theta)  # 预测值计算（dot为矩阵乘法） hΘ = xΘ₁ + Θ₂
        errors = predictions - y  # 误差计算

        gradient = (1 / m) * x.T.dot(errors)  # 梯度计算
        theta = theta - learning_rate * gradient  # 梯度下降

        cost = (1 / (m * 2)) * np.sum(errors ** 2)  # 代价函数计算
        cost_history.append(cost)

        if i % 100 == 0 or i == 1000:
            print(f'第{i + 1}次梯度下降，代价函数：{cost}')

    return theta, cost_history


m = len(x_data)
DATA = np.c_[np.ones(m), X, Y]  # 添加偏置列向量1

theta = np.zeros(DATA.shape[1])  # 参数矩阵，学习率，迭代次数
learning_rate = 0.02
iterations = 1001

theta, cost_list = gradient_descent(DATA, Z, theta, learning_rate, iterations)
print(f"最终的参数为{theta}")

x_values = np.linspace(min(x_data), max(x_data), 50)              # 等间距拟合平面数据准备
y_values = np.linspace(min(y_data), max(y_data), 50)              # 生成50个等间距XY点
X_mesh, Y_mesh = np.meshgrid(x_values, y_values)
X_std = (X_mesh - np.mean(x_data)) / np.std(x_data)                    # 标准化处理以计算Z
Y_std = (Y_mesh - np.mean(y_data)) / np.std(y_data)
Z_pred_std = theta[0] + theta[1] * X_std + theta[2] * Y_std            # 计算预测值Z(标准化后)
Z_pred = Z_pred_std * np.std(z_data) + np.mean(z_data)                 # Z的反标准化

fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={})

ax[0] = fig.add_subplot(121, projection='3d')                          # 图一(数据和拟合直线)
ax[0].scatter(x_data, y_data, z_data, c='r', marker='x', s=10)         # 原始数据点阵图
ax[0].plot_surface(X_mesh, Y_mesh, Z_pred, alpha=0.5, color='blue')    # 拟合平面
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_zlabel("Z")


ax[1].plot(range(iterations), cost_list)                                # 损失函数
ax[1].set_xlabel('Iterations')
ax[1].set_ylabel('Cost (J)')
ax[1].set_title('Cost Function Convergence')


# 创建一个 θ₁ 和 θ₂ 的网格，用于计算每对 θ 的损失
theta1_vals = np.linspace(-1, 1, 100)  # θ₁取值范围
theta2_vals = np.linspace(-1, 1, 100)  # θ₂取值范围
T1, T2 = np.meshgrid(theta1_vals, theta2_vals)

J_vals = np.zeros_like(T1)

# 固定 θ₀，计算每对 (θ₁, θ₂) 的损失值 J
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta_temp = np.array([theta[0], T1[i, j], T2[i, j]])  # 当前θ组合
        predictions = DATA.dot(theta_temp)
        errors = predictions - Z
        J_vals[i, j] = (1 / (2 * m)) * np.sum(errors ** 2)  # 损失函数




# 绘制3D损失函数图
fig2 = plt.figure(figsize=(12, 5))

# 3D 曲面图
ax3d = fig2.add_subplot(121, projection='3d')
ax3d.plot_surface(T1, T2, J_vals, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('θ₁')
ax3d.set_ylabel('θ₂')
ax3d.set_zlabel('Cost J')
ax3d.set_title('Cost Function Surface')

# 等高线图
ax2d = fig2.add_subplot(122)
contour = ax2d.contour(T1, T2, J_vals, levels=50, cmap='viridis')
ax2d.set_xlabel('θ₁')
ax2d.set_ylabel('θ₂')
ax2d.set_title('Cost Function Contour')
fig2.colorbar(contour)

plt.show()
