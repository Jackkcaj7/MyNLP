#搭建RNN模型训练MNIST手写体数据集实现分类 并用tensorboard动态跟踪观测
#保存和加载模型
import torch
import torch.nn as nn
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

train_data = KMNIST("./KMNIST_datasets",train=True,
                    
                    download=True,transform=ToTensor())
test_data = KMNIST("./KMNIST_datasets",train=False,
                   download=True,transform=ToTensor())

LR = 0.01
BATCH_SIZE = 8
epochs = 10

train_dl = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dl = DataLoader(test_data,batch_size=BATCH_SIZE)

class rnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size = 28,
            hidden_size = 50,
            bias = True,
            batch_first = True,
            num_layers = 2
        )
        self.linear = nn.Linear(50,10)
        #还可在RNN后面接其他网络层
    def forward(self,X):
        output,h_t = self.rnn(X)
        y_hat = self.linear(output[:,-1,:])
        return y_hat  #h_t[-1]=output[:,-1,:]



if __name__ == "__main__":
    writer = SummaryWriter()

    model = rnnModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),LR)

    model.train()
    for i in range(epochs):
        for data,target in train_dl :
            y_hat = model(data.squeeze()) #data shape(64,1,28,28)
            #data不符合rnn运算维度要求 所以去掉1 data.squeeze()
            loss = loss_fn(y_hat,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"epoch:{i + 1},loss{loss.item()}")
        writer.add_scalar("KMNIST_loss",loss.item(),i * len(train_dl))
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data,target in test_dl:
            y_hat = model(data.squeeze())
            _,pred = torch.max(y_hat,1)
            correct += (pred == target).sum().item()
            total += target.shape[0]

            print(f"acc:{correct / total * 100}%")
    
    writer.close()

#模型保存 加载
    #方式1
    torch.save(model,"RNN_calssifier.pth")
    model = torch.load("RNN_calssifier.pth")
    #方式2
    torch.save(model.state_dict(),"rnn_state.pth")
    model = rnnModel()
    model.load_state_dict(torch.load("rnn_state.pth"))