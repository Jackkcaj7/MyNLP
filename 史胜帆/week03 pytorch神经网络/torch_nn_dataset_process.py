# 用FashionMNIST数据集实现一个神将网络模型的训练

# 数据加载 处理
import torch
from torchvision.datasets import FashionMNIST #导入数据集所在包
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import ToTensor #把图像转张量

#加载数据集到本地 
train_data = FashionMNIST(root='./fashionMNIST_data',train = True,
                          download = True,transform=ToTensor())
# train = True 表示下载的是训练集
test_data = FashionMNIST(root='./fashionMNIST_data',train = False,
                         download = True,transform=ToTensor()   )
# train = False表示下载的是测试集
# 加载过程pytorch会自动检测root地址是否有被下载的数据集 有则不下载

# #观察下载完成的train_data
# print(train_data,test_data)
# #发现是一个Dataset对象 60000条训练样本 10000条测试样本
# print(train_data[0])
# img,clzz = train_data[1]
# print(img,clzz) #img是图像 PIL.Image是对象 是python的原始数据类型
# # clzz是类别 是一个数 
# #用matplotlib观察图像
# plt.imshow(img,cmap = 'gray')
# plt.title(clzz)
# plt.show() # 是一张28*28的上衣略缩图

# 现在是图像 如何把图像数据集拿来训练模型 -> 图像转tensor
# torchvision.transforms.v2.ToTensor 
# 使用方式： 在加载数据时在FasionMNIST()中追加参数 transform=ToTensor() 
#得到张量矩阵
# 把原来查看train_data的方式注释掉 查看totensor后的数据
print(train_data[0]) # 返回一个元组 第一元素是图像的tensor数据 第二是标签
img_tensor ,clzz = train_data[0]
print(img_tensor.shape,clzz) # img_tensor是图像数据 (1个颜色通道,高,宽)
#要训练 就要把一个图像转成一个1维张量
print(img_tensor.reshape(-1).shape) # shape(784) 实际是1个行向量

# 要观察到底是级分类问题 要观察标签类别有多少种
# 把train_data 中的clzz类别全收集起来 set集合去重观察
labels = set([clzz for img_tensor,clzz in train_data])
print(labels) # 0-9 共10个类别

# 数据准备好了 开始搭建模型准备训练