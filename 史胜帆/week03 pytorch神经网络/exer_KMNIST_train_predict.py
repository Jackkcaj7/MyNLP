# 使用ptorch搭建神经网络模型 实现对KMNIST数据集的训练
# 改变神经元个数 隐藏层数量 batch_size 学习率 观察模型的准确率变化
import torch
import torch.nn as nn
from torchvision.datasets import KMNIST
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#加载KMNIST数据 
train_data = KMNIST('./KMNIST_datasets',train=True,
                    download = True,transform=ToTensor())
test_data = KMNIST('./KMNIST_datasets',train=False,
                   download=True,transform=ToTensor())
# print(train_data,test_data)
# print(train_data[0],test_data[0])
# # 观察发现 train_data 有60000样本 test_data有10000样本
# # 每个样本是（img.class） 
# img,clzz = train_data[0]
# print(img)
# plt.imshow(img,cmap='gray')
# plt.title(clzz)
# plt.show()
# # pyplot可视化展示图像 发现是手写字的略缩图 
# # PIL.Image python原始数据类型 是一张图像 大小（1，28，28）

# Totensor转张量 仍然是元组（img_tensor,class）
img_tensor,clzz = train_data[0]
print(img_tensor,clzz)
print(img_tensor.shape) # shape(1,28,28)
# 观察有几类 确定是几分类问题 从而确定输出层神经元个数
labels = set([clzz for img_tensor,clzz in train_data])
print(labels) # 0-9 有10类样本

#模型参数
# 1权重参数 在搭建的nn网络中会自动生成 不用手动随机初始化权重参数
# 2超参数
LR = 0.01
epochs = 30
BATCH_SIZE = 128

# 60000个样本数据batch切分
train_dl = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

# nn网络各层参数
# X输入层 shape(,784)
# 输入到隐藏层 W参数 shape(784,128)
# 输入到隐藏层 bias参数 shape(128,)
# 隐藏到输出层 W shape(128,10)
# 隐藏到输出层 bias shape(10,)
# 输出层输出结果Y shape(,10)

# 搭建nn模型
model = nn.Sequential(
    nn.Linear(784,128),
    nn.Sigmoid(),
    nn.Linear(128,10)
)

#损失函数和优化器
loss_fn = nn.CrossEntropyLoss() # 多分类采用交叉熵损失
optimizer = torch.optim.SGD(model.parameters(),LR)

#模型训练
for i in range(epochs):
    for data,target in train_dl:
        y_hat = model(data.reshape(-1,784))
        loss = loss_fn(y_hat,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"loss:{loss.item()}")

# 模型推理预测
test_dl = DataLoader(test_data,batch_size=BATCH_SIZE)
correct = 0
total = 0
for data,target in test_dl:
    y_hat = model(data.reshape(-1,784))
    _,predicted = torch.max(y_hat,dim = 1)
    correct += (predicted == target).sum().item()
    total += target.shape[0]

print(f"acc:{correct / total * 100}%")

