## 1.用pytorch框架实现逻辑回归模型训练 数据集用iris鸢尾花 取其前2类样本数据 用sigmoid函数映射到01区间完成二分类任务
## 2.用pytorch搭建神经网络NN 实现对FashionMNIST数据集的训练
#### 2.1 torchvision.datasets.FashionMNIST 时装略缩图数据集加载到本地
#### 2.2 torchvision.transform.v2.ToTensor 把（1，28，28）图像转成tensor
#### 2.3 torch.utils.data DataLoader 实现batch切分
#### 2.4 神经网络由线性层和sigmoid函数构成
#### 2.5 分batch_size 训练加快速度
#### 2.6 模型推理预测阶段计算准确率
## 3.练习：用pytorch搭建神经网络NN 实现对KMNIST手写体数据集的训练
## 调整学习率lr NN的神经元个数 隐藏层层数 和batch_size大小观察模型推理阶段准确率的变化
