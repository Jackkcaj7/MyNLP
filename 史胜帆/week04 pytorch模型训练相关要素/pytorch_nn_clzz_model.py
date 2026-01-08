# 用class类的方式实现一个神经网络的搭建 不用nn.Sequential()的方式
# 这样类的方式构建模型的方式更普遍 标准
# 这样写的优势在于 可以对模型结构灵活变更和定制


import torch
import torch.nn as nn

class TorchNN(nn.Module): # 继承自nn.Module模块
    #初始化
    def __init__(self): # self指代正在调用的对象表示当前对象实例化后对象它自己 
        super().__init__() # 必写 继承父类的属性和方法
        # 自定义模型内容的部分
        self.linear1 = nn.Linear(784,512)
        self.bn1 = nn.BatchNorm1d(num_features = 512) #对shape(batch,512)做归一化
        self.linear2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(num_features = 128)
        self.linear3 = nn.Linear(128,10)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p = 0.3)
    # forward前向运算 是nn.Module中forward方法的重写
    def forward(self,input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        final =self.linear3(out)
        
        return final

# 模型测试
if __name__ == '__main__':  #这行话去掉 当前整个文件仍然正常运行
    # 这行话的目的和作用 防止这个文件的模块被调用的时候下面这些测试内容也被运行

    model = TorchNN() # 对象实例化 创建对象 self就指model自己
    input_data = torch.randn(10,784)
    final = model(input_data) 
    # 在调用过程中 pytorch就会自动记录节点生成计算图
    print(final.shape)
    # 尝试模型调试 以后写程序就不能是去抄 
    # 遇到不会的写错的自己调试完成直至成功运行 跟踪观察纠正

#这样定义NN模型就可以在其他文件里调用这个文件的模型
# from pytorch_nn_clzz_model import TorchNN
# model = TorchNN()
# 这样替换模型的过程也方便多了

# 模型训练前
model.train()
#模型推理预测阶段
model.eval()
