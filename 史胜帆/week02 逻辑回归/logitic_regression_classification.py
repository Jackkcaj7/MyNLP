# 用逻辑回归完成分类任务
# 调用机器学习库 scikit-learn 获取分类数据 用于训练推理 数据类型是np.ndarray
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


# # 拿到数据先看看数据长什么样 内容 大小
# X,y = make_classification()
# print(X,y) # y 是伯努利分布的k取值即y_i 0或1 符合逻辑回归数据要求
# print(X.shape,y.shape)
# print(X[0],y[0])

# 生成分类数据
X,y = make_classification(n_samples = 150,n_features = 10)

# 对于训练数据 我们不可能获取到所有可能的训练数据
# 采用局部训练样本训练模型 模型只会局部训练样本的任务 之外的任务不会 
# 导致训练误差和测试误差特别大 即训练表现好 测试表现差 就是过拟合模型
# 其中对于局部训练样本之外的样本数据模型表现不好 就成为模型泛化性差

#数据拆分 train test 用sklearn中的train_test_split() 按7：3整数拆分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
print(X_train.shape,X_test.shape)

#随机初始化模型参数
#1权重参数
theta = np.random.randn(1,10) # 含10个元素的行向量 10个参数对应模型10个特征
bias = 0
#2超参数
lr = 0.01
epochs = 3000

#前向传播
def forward(X,theta,bias):
    #线性运算
    z = np.dot(theta,X.T) + bias 
    #必须是theta @ X.T 才能保证每个参数对应每个样本的特征对齐相乘再相加
    # 这样完成的矩阵运算 简化计算 比循环完成每个样本和参数之间相乘相加快的多
    # 激活函数 非线性因素 二分类 sigmoid
    y_hat = 1 / (1 + np.exp(-z)) #激活函数不进行维度变换
    return y_hat

# 损失函数
def loss(y,y_hat):
    epsilon = 1e-8 # 定义极小值避免逻辑回归损失函数伯努利分布的似然函数log真数 = 0
    return -y*np.log(y_hat + epsilon) - (1-y)*np.log(1-y_hat + epsilon)


# 计算梯度
def calc_gradient(X,y,y_hat):
    m = X.shape[0]
    delta_theta = np.dot((y_hat - y),X) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta,delta_bias

# 模型训练
for i in range(epochs):
    y_hat = forward(X_train,theta,bias)
    loss_val = loss(y_train,y_hat)
    delta_theta ,delta_bias = calc_gradient(X_train,y_train,y_hat)
    theta = theta - lr*delta_theta
    bias = bias - lr*delta_bias

    if i % 100 == 0:
        acc = np.mean(np.round(y_hat) == y_train)
        print(f"epoch:{i},loss:{np.mean(loss_val)},acc:{acc}")


# 模型推理 （预测）

idx = np.random.randint(len(X_test))
x,y = X_test[idx],y_test[idx]
predicted_y = np.round(forward(x,theta,bias))
print(f"predicted_y:{predicted_y},real_y:{y}")



# 逻辑回归的应用：意图识别（行为预测）、情感分析（正负情感）、金融交易（涨或跌）
# 二分类判断 都用的上逻辑回归