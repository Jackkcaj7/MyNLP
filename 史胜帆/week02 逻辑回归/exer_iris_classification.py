# exercise 练习
# 使用sklearn中 iris鸢尾花数据集实现逻辑回归二分类任务 
# path：scikit-learn -> API -> datasets -> load_iris 里面userguide了解数据集更多信息
#调整学习率,样本数据train/test拆分比率  观察训练结果
# 把训练后的模型参数保存到一个文件中 在另一个代码文件中调用模型参数实现推理预测

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

# #查看数据
# X,y = load_iris(return_X_y = True)
# print(X,y)
# print(X.shape,y.shape)
# print(X[0],y[0])

X,y = load_iris(return_X_y = True)
X,y = X[:100],y[:100]
# print(X.shape,y.shape)
# print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
# print(X_train.shape,y_train.shape)

# 定义模型参数
# 1随机初始化权重参数
theta = np.random.randn(1,4)
bias = 0
# 2超参数
lr = 0.01
epochs = 30

#前向传播
def forward(theta,X,bias):
    #线性函数
    z = np.dot(theta,X.T) + bias # (1,70)
    # 激活函数
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 损失函数 逻辑回归 伯努利分布的似然函数取log加-号
def loss(y,y_hat):
    epsilon = 1e-8 # 设定极小值防止log真数为0
    return -y*(np.log(y_hat + epsilon)) - (1-y)*(np.log(1 - y_hat + epsilon))

# 计算梯度
def calc_gradient(X,y,y_hat):
    m = X.shape[0]
    delta_theta = np.dot((y_hat - y),X) / m # 公式来源于求导结果
    delta_bias = np.mean(y_hat - y) #求导结果
    return delta_theta,delta_bias

# 训练模型
for i in range(epochs):
    y_hat = forward(theta,X_train,bias)
    loss_val = loss(y_train,y_hat)
    delta_theta,delta_bias = calc_gradient(X_train,y_train,y_hat)
    theta = theta - delta_theta
    bias = bias - delta_bias

    acc = np.mean(np.round(y_hat) == y_train)
    print(f"epoch:{i},loss:{np.mean(loss_val)},acc:{acc}")

# 模型参数和测试集数据的保存
np.save('theta.npy',theta)
np.save('bias.npy',bias)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)