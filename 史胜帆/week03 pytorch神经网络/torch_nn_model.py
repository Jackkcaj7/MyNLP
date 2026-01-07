# 搭建神经网络模型 准备训练FashionMNIST数据集

import torch.nn as nn # 导入nn的所有常用模块
import torch

# 搭建模型之前 先搞清楚整个模型运行过程中的维度变化
# 包括 输入层 隐藏层(W参数 bias参数) 输出层 的维度大小
# X输入层 shape(60000,784) 即(,784)
# 输入层->隐藏层 shape(784,128) W参数 其中 128可理解为神经元个数 由人为指定 不是固定的
# 输入层->隐藏层 shape(128,) bias参数 隐藏层的bias神经元是1个 在nn图中在输入层最下面 128个元素分别与X@W.T结果中的128个y_i相加
# 隐藏层->输出层 shape(128,10) W参数 其中 10是固定的 因为数据集有10类样本 最终要预测10类 因此必须输出10个神经元对应10个概率值
# 隐藏层->输出层 shape(10,) bias参数 输出层的bias神经元是1个 在nn图中在隐藏层最下面 相加原理同隐藏层bias
# 输出层的输出Y shape(,10) # 10个类别

#搭建模型
model = nn.Sequential(
    nn.Linear(784,128),
    nn.Sigmoid(),
    nn.Linear(128,10)
)


#损失函数和 优化器
loss = nn.CrossEntropyLoss() # 多分类用交叉熵损失 作用 计算损失 计算梯度
# 优化器 作用 模型参数更新 收集梯度 清空梯度 不仅仅是theta - lr*delta_theta这么简单的参数更新
# 优化器在torch.optim里
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01) 
# model.parameters() 收集模型所有参数 用于参数更新
