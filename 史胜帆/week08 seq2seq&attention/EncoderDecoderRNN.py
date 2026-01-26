#实现基于RNN的Encoder Decoder模型

import torch
import torch.nn as nn

#Encoder
class Encoder(nn.Module):
    def __init__(self,input_dim,ebd_dim,hidden_dim,dropout):
        super().__init__()
        self.ebd =nn.Embedding(input_dim,ebd_dim)
        self.rnn = nn.GRU(ebd_dim,hidden_dim,dropout=dropout,
                            batch_first=True,bidirectional=True)
        

    def forward(self,token_seq_metric):
        embed = self.ebd(token_seq_metric)
        output,h_t = self.rnn(embed)
        #Encoder 返回最后一个时间步的隐藏状态 即seq2seq中的c 抽象概要
        #双向h_t结果 可以拼接或相加
        #拼接
        return torch.cat(h_t[0],h_t[1],dim=1) #在h_t[i]的第1个维度做拼接 [batch,hidden_size*2]
        #相加 在第0个维度做相加 即在2维情况下的h_t[i]的行上
        #return h_t.sum(dim=0) 

#Decoder
class Decoder(nn.Module):
    def __init__(self,input_dim,ebd_dim,hidden_dim,dropout):
        super().__init__()
        self.ebd = nn.Embedding(input_dim,ebd_dim)
        self.rnn = nn.GRU(ebd_dim,hidden_dim*2,batch_first=True,dropout=dropout)
        # Enocoder 和Decoder要使用相同的运算模型才可以正确编解码
        # Decoder 不需要获取语义信息只用输出预测结果 因此不需要bidirectional
        # 由于Encoder使用的是cat拼接输出维度是hidden*2 
        # 那么decoder的rnn的h_0也是hidden*2 那么Decoder的rnnde1接收维度也应该*2
        #如果用的是加sum 就还是hidden_dim
        # Decoder的输入只有1部分 Encoder的抽象概要c(长期记忆状态)  
        # 此处输入文本input就可以理解为中译英中的中文句子输入等待被翻译
        self.fc = nn.Linear(hidden_dim*2,input_dim)  #解码词典中词汇概率所构成的向量
        #input_dim的大小等于整个词汇表长度 fc过后得到的是词典大小个概率值每个概率对应一个词的概率
        # 从中选一个最大概率对应的词token就是最终预测结果 这是为后续得到翻译结果token序列做准备
        #是input_dim的原因 
        # 涉及Decoder原理：根据decoder中前一个token的输出预测下一个token的概率
        #解码器的输入要比输出早一个时间步
        #即 [<BOS>] [how] [are] [you] [?]
        #   [how]     [are] [you] [?]   [<EOS>]
        #其中<BOSt> <EOS>分别是起始标志和结束标志 是EncoderDecoder正确工作的保证
        
        #整个Decoder过程相当于不断用decoder前一个输出的token预测下一个token 直到预测到<end>结束
        # 也就是说如果一直不遇到end decoer结果就会无限长
    def forward(self,input,c):
        ebd = self.ebd(input)
        output,h_t = self.rnn(ebd,c.unsqueeze(0)) #c是rnn族里的h_0初始状态
        #这里既然是Decoder 那么h_0就是c (num_layers*bidirection,batch,hidden_size)
        output = self.fc(output) #得到的是解码序列矩阵 [batch,seq_len,input_dim] 
        #最终解码出来的是seq_len个token input_dim也等于vocab_len
        #每一个seq都是维度大小等于vocab_len的概率值向量

        #如此 解决了输入输出不对齐 比如文本翻译 即RNN族无法解决的 N vs M 问题
