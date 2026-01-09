# 练习 对sklearn.datasets.fetch_olivetti_face数据集做训练并预测类别

from sklearn.datasets import fetch_olivetti_faces
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#数据预处理 数据加载 数据观察 数据整理
# X,y = fetch_olivetti_faces(data_home='./fetch_olivetti_faces_data',shuffle=True,return_X_y=True)
# print(X.shape,y.shape)
# print(X,y)
# 观察发现 使用return_X_y 返回的是X,y样本数据和标签的ndarray数组

#通过查看sklearn 中这一数据集的详情 知道它有data target images三个属性
faces = fetch_olivetti_faces(data_home='./fetch_olivetti_faces_data',shuffle=True)
# print(faces.data.shape)
# print(faces.target.shape)
# print(faces.images.shape)

# plt.imshow(faces.images[0],cmap='gray')
# plt.show()
# # 观察发现 faces数据集有400个样本 每个样本4090特征和1个标签
# #每个样本对应一张(1,64,64)的图片 是人脸图片

# #查看有多少类样本 确定是几分类问题 确定NN输出维度
# labels = set([tagt for tagt in faces.target])
# print(labels) #0-39 共40类样本

# 这里不采用直接用return_X_y获取X,y的方式 
#采用对faces.data faces.target组合成数据集的方式
X,y = faces.data,faces.target
#train test 数据切分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
print(X_train.shape,y_train.shape)
#把X_train y_train整合成trian_data(X_train,y_train) 用元组实现
train_data = [(x,y) for x,y in zip(X_train,y_train)]

#模型参数
#1权重参数 会自动在NN进程中生成
#2超参数
LR = 0.01
epochs = 30
BATCH_SIZE = 30

#DataLoder batch切分
train_dl = DataLoader(train_data,batch_size=BATCH_SIZE)

#NN维度变换
# (,4096)
# (4096,2048)
# (2048,512)
# (512,128)
# (128,40)
# (40,)

#模型搭建
class Model(nn.Module):
    #初始化
    def __init__(self):
        super().__init__()
        #模型结构
        self.linear1 = nn.Linear(4096,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.linear2 = nn.Linear(2048,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128,40)
        self.drop = nn.Dropout(p = 0.3)
        self.act = nn.ReLU()
    
    def forward(self,input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.drop(out)
        out = self.act(out)
        y_hat = self.linear4(out)
        
        return y_hat

model = Model()

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),LR)

if __name__ == '__main__':
    
    model.train()
    #模型训练
    for i in range(epochs):
        for data,target in train_dl:
            
            y_hat = model(data.reshape(-1,4096))
            loss = loss_fn(y_hat,target.long())
            #交叉熵损失函数要求标签（target）必须是torch.long(64位整数)类型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"epoch:{i + 1},loss:{loss.item()}")

    #模型推理
    model.eval()
    test_data = [(data,target) for data,target in zip(X_test,y_test)]
    test_dl = DataLoader(test_data,batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    with torch.no_grad():
        for data,target in test_dl:
            y_hat = model(data.reshape(-1,4096))
            _,predicted = torch.max(nn.functional.softmax(y_hat,dim = -1),dim=1)
            correct += (predicted == target).sum().item()
            total += target.shape[0]
        
        print(f"acc:{correct / total * 100}%")