# 梯度下降法的演示
import matplotlib.pyplot as plt
import numpy as np

plot_x = np.linspace(-1,6,150)
plot_y = (plot_x - 2.5) ** 2 - 1
plt.plot(plot_x,plot_y)
plt.show()

# 定义损失函数
def loss(theta):
    return (theta - 2.5) ** 2 - 1

#定义导数（梯度）
def derivate(theta):
    return 2 * (theta - 2.5)

#设定模型参数 
theta = 0.0 #随机初始化模型参数
eta = 0.1 # 学习率
epsilon = 1e-8 # 迭代阈值

theta_history = [theta]  # 为了后续绘制梯度下降过程的图形 记录theta所有值

while True:
    last_theta = theta  # 记录上一次theta 为了计算上一次损失值
    gradient = derivate(theta) # 参数值对应的梯度
    theta = theta - eta * gradient # 参数更新
    theta_history.append(theta)
    loss_value = loss(theta) # 计算损失值
    if abs(loss(last_theta) - loss_value) < epsilon:
        break

print(theta,loss_value)

plt.plot(plot_x,plot_y)
plt.plot(theta_history,loss(np.array(theta_history)),color = 'r',marker = 'd')
plt.show()

# 查看训练次数（梯度下降次数）

print(len(theta_history) - 1)

# 调整学习率eta 变大变小 观察梯度消失 震荡和爆炸
# 学习率大 造成加速收敛 震荡 错过全局最优点 梯度爆炸 
# 学习率小 收敛速度缓慢 收敛到局部最优点 
#可以把上述梯度下降过程 图像绘制过程封装成函数 避免重复造轮子




