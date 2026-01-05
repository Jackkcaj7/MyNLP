# 使用iris数据集训练逻辑回归模型后得到的参数和测试数据集进行模型推理预测
import numpy as np

# 加载已保存的参数和测试数据集 
# 如果参数文件和当前推理代码文件不在同一个文件夹下 要写绝对路径
theta = np.load('theta.npy')
bias = np.load('bias.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# 用于推理预测的前向运算
def forward(theta,X,bias):
    z = np.dot(theta,X.T)
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

idx = np.random.randint(len(y_test))
x,y = X_test[idx],y_test[idx]

y_predicted = np.round(forward(theta,x,bias))
print(f"y_predicted:{y_predicted},y_true:{y}")