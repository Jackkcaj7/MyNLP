#
import torch

rnn = torch.nn.RNN(
    input_size = 28,#输入特征维度
    hidden_size = 50, #隐藏层神经元个数 w_ih shape(50,28) w_hh shape(50,50)
    #(50,28)的原因是 在理论上应该是(28,50) 
    # 但这是在pytorch框架中 线性运算公式是y_hat = x@w.T 
    # 所以这里的w应该是w.T.T=w shape(50,28)
    bias = True,
    batch_first=True
)
print(rnn)

#创建输入数据
#shape(batch,times,features)
X = torch.randn(10,28,28) # 10个样本 每个样本28个时间步 每个时间步28个特征
outputs,last_time_output = rnn(X)
print(outputs.shape,last_time_output.shape)


#batch_first前 
# input(times,batch,in_features) 
# out1(times,batch,hidden_features) 
# out2(layers*directions,batch,hid_features) 
#outputs 所有时间步的输出的拼接 shape(10,28,50)
#last_time_output 最后一次时间步的运算结果 shape(1,28,50)

#batch_first后
# input(batch,times,in_features) 
# out1(batch,times,hidden_features) 
# out2(layers*directions,batch,hid_features) 
#outputs shape(10,28,50)
#last_time_output shape(1,10,50)
#可见 在pytorch中无论是否令输入的维度batch_first out2的维度含义都不变
#最后一个时间步保留了前面所有时间步的记忆

#只有在单向单层RNN的特定条件下，
# output[:, -1, :]才等于h_n（去掉 h_n 多余的第一个维度后），
# 在其他情况下两者是不同的。
