# 用pytorch实现iris鸢尾花数据集的逻辑回归模型
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y = True)
X,y = X[:100],y[:100]

# 与用numpy不同的是 这些数据此时都还是一般python数据类型 
# 用pytorch的前提是先把数据集转变为张量tensor
# 创建张量数据集
tensor_x = torch.tensor(X,dtype = torch.float32)
tensor_y = torch.tensor(y,dtype = torch.float32)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# 模型参数
# 1权重参数
w = torch.randn(1,4,requires_grad = True) 
# requires_grad 揭示了tensor的第3个特性 梯度 默认情况下w.grad = None
# w.grad 保存的是参数的梯度值 这样就把参数和其梯度捆绑一起
b = torch.randn(1,requires_grad = True)
# 2超参数
lr = 0.01
epochs = 300

#模型训练
for i in range(epochs):
    # 前向传播
    z = torch.nn.functional.linear(tensor_x,w,b) 
    # y = x @ w.T + b 其中w默认行向量
    z = torch.sigmoid(z)  # shape(100,1)]  映射到0 1区间 最优超平面是线性的
    

    # 损失函数
    loss = torch.nn.functional.binary_cross_entropy(z.reshape(-1),tensor_y,reduction ='mean')
    # print(z.shape,tensor_y.shape)

    #计算梯度
    loss.backward() # 用计算图的思维理解这一步 
    # 从tensor_x tensor_y到loss的每一步计算都在计算图中形成节点并连接起来
    # 形成一个从输入到输出的计算流程图
    # backward就可以按照这一计算图对每个参数按流程求导计算梯度
    # 梯度值和参数w捆绑在一起 保存在w.grad b.grad中

    # 参数更新
    with torch.no_grad(): # 关闭梯度计算跟踪
        w -= lr * w.grad
        b -= lr * b.grad

        # 用完梯度之后要手动清空梯度 pytorch不自动清空梯度
        w.grad.zero_() #zero_()是inplace方法 表示在w.grad自身更改 w.grad还在 = None
        b.grad.zero_()
    print(f"loss:{loss.item()}")
