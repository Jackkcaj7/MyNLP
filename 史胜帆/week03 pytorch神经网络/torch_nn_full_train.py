# FashionMNIST 数据集完整训练

import torch
from torchvision.datasets import FashionMNIST 
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import ToTensor 
import torch.nn as nn
from torch.utils.data import DataLoader

#加载数据集到本地 
train_data = FashionMNIST(root='./fashionMNIST_data',train = True,
                          download = True,transform=ToTensor())
test_data = FashionMNIST(root='./fashionMNIST_data',train = False,
                         download = True,transform=ToTensor()   )

#模型参数
# 1 权重参数 已在模型创建中自动生成
# 2 超参数
LR = 0.01
epochs =30
BATCH_SIZE = 128

#batch封装
train_dl = DataLoader(train_data,batch_size = BATCH_SIZE,shuffle = True)
# shuffle 在封装过程中打乱数据 一定程度上有利于提高模型泛化性

# 搭建模型
model = nn.Sequential(
    nn.Linear(784,128),
    nn.Sigmoid(),
    nn.Linear(128,10)
)


# 损失函数和 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),LR)


#训练模型
for i in range(epochs):
    for data,target in train_dl:
        # 前向传播
        y_hat = model(data.reshape(-1,784)) 
        # 写-1的好处 避免60000数据不能被batch_size整除
        # 计算损失
        loss = loss_fn(y_hat,target)
        #反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 计算梯度 每个参数的梯度结果和optimizer中的参数捆绑在一起 W.grad
        optimizer.step() # 用梯度参数更新

    print(f"loss:{loss.item()}")

# 这样的训练方式要进行60000个样本的训练多轮
# 太慢了 怎么办 把60000个样本一起装到内存里去训练？ 如果图片较大 不可能
# 介于 单个样本训练和整体一次性训练 之间 的折中办法 batch 
# 采用batch一批次中含多个样本 也不会一个整体那么多 
# 1带入数据加载器 对60000个数据封装 torch.utils.data.DataLoader
# 2在超参数中定义批次参数 batch_size
# 3在数据加载之后 搭建模型之前 对train_data进行batch封装
# 4DataLoader之后 
# 4.1数据自动转为tensor 则去掉torch.tensor(target)
# 4.2引入了batch 则一个batch一个batch以矩阵形式运算 则 data 要改变形状为
#     shape(batch_size,784)

#模型推理预测
test_dl = DataLoader(test_data,batch_size = BATCH_SIZE)

correct = 0
total = 0
with torch.no_grad():
    for data,target in test_dl:
        y_hat = model(data.reshape(-1,784)) #y_hat shape(batch,10)
        _,predicted =torch.max(y_hat,dim = 1) # 返回一个batch中的每行中最大值及对应索引
        total += target.shape[0] # test_dl的样本总数
        correct += (predicted == target).sum().item() # 预测正确总数

    print(f"acc:{correct / total * 100}%")