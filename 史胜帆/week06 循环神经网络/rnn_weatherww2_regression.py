#使用RNN实现一个天气预测模型 预测1天和连续5天的最高气温
#数据集使用 weather_world_war2 summary of weather.csv
#用tensorboard跟踪观测训练过程


#预处理
#1下载数据
#观察发现是.csv格式文件 文件包含多个属性列 样本数量较大约10万条 
# 样本涉及1940-1945年全球不同气象站的天气信息 每个气象站监测总天数不同

#2从原始文件中提取要用的关键数据
#3观察数据 去除异常值 用plot绘图观察 
# 去除气温天数过少的站点的样本 去除每个站点极寒极热异常气温值
#好数据是好模型的一半 好数据要在类别标签上有明确区分度 输入特征上不异常
#4思考模型输入需求 和 输出目标
#5 构建符合模型输入需求的数据集 包括从已有数据中提取出输入特征和标签 构成数据集
#要注意的是 构建数据集时要具有随机性
#6 模型训练流程

import csv
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

#下载数据 
#提取关键数据
#初始化字典 存放数据
sta_temp = {}
with open("Summary_of_Weather.csv","r",encoding="UTF-8") as f:
    reader = csv.DictReader(f)
    for line in reader:
        sta = line['STA']
        sta_temp[sta] = sta_temp.get(sta,[])
        sta_temp[sta].append(float(line['MaxTemp']))
    
print(f"气象站总数：{len(sta_temp)}") #159
#用plot绘制数据 观察 去除异常值
#先观察整体 每个气象站的数据量 去除数据量极少的站点 不参与样本
temps_len = [len(temps) for temps in sta_temp.values()]
plt.bar(range(len(sta_temp)),temps_len)
plt.show()
#发现有些站点的数据量是个位数 
#后续训练中 以20天天气预测第21天 所以去掉数据量小于20的站点
#并且 此时已经不再需要站点信息 因此只收集每个站点的maxtemp
max_temps =  [temps for temps in sta_temp.values() if len(temps) > 20]
print(f"保留站点数：{len(max_temps)}") #158
#用subplot绘制158个子图 每个子图的y轴是温度数值
plt.figure(figsize = (20,200)) #建立一个20*200的图形用来绘制子图
# 子图按照79*2布局 每行2个
for i in range(79):

    temps_l,temps_r = max_temps[2*i],max_temps[2*i+1]
    plt.subplot(79,2,i*2+1)
    plt.plot(range(len(temps_l)),temps_l)
    plt.subplot(79,2,2*i+2)
    plt.plot(range(len(temps_r)),temps_r)
plt.show()
#观察发现同一气象站中有温度严重偏离其他温度值
#这是异常值 不利于训练 去掉
#过滤掉每个站点中的异常值 设定<-17度的是异常气温值
max_temps = [[temp for temp in temps if temp > -17] for temps in max_temps]

#构建数据集 
# 提取 拿出随机站点随机起始位置n_steps个温度数值 并转成ndarray float32类型 
# 封装 
# batch切分
def generate_time_series(temp_values,batch_size,n_steps):
    #生成数据集矩阵 用于后续填充
    series = np.zeros((batch_size,n_steps))
    #生成随机样本索引 用于后续提取随机站点数据
    sta_idx = np.random.randint(0,len(temp_values),batch_size)
    #提取batch_size份 每个随机站点的任意起始位置的n_steps温度数值
    for i,idx in enumerate(sta_idx):
        temp_start = np.random.randint(0,len(temp_values[idx])-n_steps)
        series[i] = np.array(temp_values[idx][temp_start:temp_start+n_steps])
    # X[batch_size,n_steps-1,1] y[batch_size,1]
    return series[:,:n_steps,np.newaxis].astype(np.float32),series[:,-1,np.newaxis].astype(np.float32)

#调用generate_time_series生成训练验证测试集
n_steps = 21
X_train,y_train = generate_time_series(max_temps,7000,n_steps)
X_valid,y_valid = generate_time_series(max_temps,2000,n_steps)
X_test,y_test = generate_time_series(max_temps,1000,n_steps)
print(X_train.shape,y_train.shape) #(7000, 21, 1) (7000, 1)

#绘制几张预测样例图形 可视化观察maxtemp时间序列
def plot_series(X,y = None,y_pred = None,y_pred_std = None,x_label = "$t$",y_label ="$y$"):
    # 设定绘制15张图 3*5
    r,c = 3,5
    #用子图数组绘制
    fig,axes = plt.subplots(nrows=r,ncols=c,sharex=True,sharey=True,figsize=(20,10))
    for row in range(r):
        for col in range(c):
            plt.sca(axes[row][col])
            ix = col + row*c #1张子图绘制1个气象站温度 ix是第几张子图也对应第几个气象站
            plt.plot(X[ix],".-") #折线图
            if y is not None:#绘制待预测点位置 用bx蓝色×绘制
                plt.plot(range(len(X[ix]),len(X[ix])+len(y[ix])),y[ix],"bx",markersize = 10)
            if y_pred is not None:#绘制预测结果 用ro 红色圆圈标记 便于观察预测准不准
                plt.plot(range(len(X[ix]),len(X[ix])+len(y_pred[ix])),y_pred[ix],"ro")
            if y_pred_std is not None:#预测标准差 量化预测结果的不确定性大小
                plt.plot(range(len(X[ix],len(X[ix])+len(y_pred[ix]))),y_pred[ix]+y_pred_std[ix])
                plt.plot(range(len(X[ix]),len(X[ix])+len(y_pred[ix])),y_pred[ix]+y_pred_std[ix])
            plt.grid(True)#添加网格线

            #设置标签所在位置
            if row == r - 1:
                plt.xlabel(x_label,fontsize = 16)
            if col == 0:
                plt.ylabel(y_label,fontsize = 16,rotation = 0)
    
    plt.show()

plot_series(X_test,y_test)

#用Dataset构建自定义张量数据集
class TimeSeriesDataset(Dataset):
    def __init__(self,X,y=None,train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,ix):
        if self.train:
            return torch.from_numpy(self.X[ix]),torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])

dataset = {
    'train':TimeSeriesDataset(X_train,y_train),
    'eval':TimeSeriesDataset(X_valid,y_valid),
    'test':TimeSeriesDataset(X_test,y_test,train=False)
}
# 对数据集batch切分
dataloader = {
    'train':DataLoader(dataset['train'],shuffle=True,batch_size=64),
    'eval':DataLoader(dataset['eval'],shuffle=False,batch_size = 64),
    'test':DataLoader(dataset['test'],shuffle = False,batch_size = 64)
}

#搭建模型
class MyModel(nn.Module):
    def __init__(self):
        self.rnn = nn.RNN(input_size=1,hidden_size=64,num_layers=1,batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        output,h_t = self.rnn(x)
        y_hat = self.fc(output[:,-1,:])

        return y_hat

#模型训练要素 模型训练及模型评估
device = "cuda" if torch.cuda.is_available() else "cpu"
def fit(model,dataloader,epochs):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    bar = tqdm(range(1,epochs+1))

    for epoch in bar:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X,y = batch
            X,y = X.to(device),y.to(device)
            y_hat = model(X)
            optimizer.zero_grad()
            loss = criterion(y_hat,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        eval_loss =[]
        with torch.no_grad():
            for batch in dataloader['eval']:
                X,y = batch
                X,y = X.to(device),y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat,y)
                eval_loss.append(loss.item())

        bar.set_description(f"train_loss:{np.mean(train_loss)},eval_loss:{np.mean(eval_loss):.5f}")

#模型预测  
def predict(model,dataloader):
    model.eval()
    with torch.no_grad():
        #创建一个张量列表存储所有预测结果
        preds = torch.tensor([]).to(device)
        for batch in dataloader['test']:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds,pred])
        
        return preds

model = MyModel()
fit(model,dataloader,epochs = 20)
y_pred = predict(model,dataloader['test'])
plot_series(X_test,y_test,y_pred.cpu().numpy())
eror = mean_squared_error(y_test,y_pred.cpu())
print(eror)

