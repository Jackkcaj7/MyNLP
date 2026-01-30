#融合attention机制的encoderdecoder模型

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,input_dim,ebd_dim,hidden_dim,num_layers=2,
                 dropout = 0.2):
        #bideriectional 让encoder挖掘更多语义信息
        super().__init__()
        self.ebd = nn.Embedding(input_dim,ebd_dim)
        self.rnn = nn.GRU(ebd_dim,hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout)
        
    def forward(self,input):
        out = self.ebd(input)
        outputs,h_t = self.rnn(out)

        return outputs[:,-1,:],outputs
     #outputs这样取就等于h_t拼接结果即c 返回的outputs用于attention计算K

class Attention(nn.Module):
    def __init__(self):#Attention没有初始化参数
        super().__init__()
    
    def forward(self,enc_rnn_output,dec_rnn_output):
        #1计算K Q关联
        # 这里enc_rnn_output (batch,seq_len1,hidden_dim)
        #dec_rnn_output (batch,seq_len2,hidden_dim)
        #因此permute
        a_t = torch.bmm(enc_rnn_output,dec_rnn_output.permute(0,2,1))
        #2关联转权重
        #要关注的是seq_token_i之间的关联 所以dim=1 不是2
        a_t = torch.softmax(a_t,dim=1)
        #3计算enc_rnn_output贡献值
        # 根据第1次计算a_t的维度变换 要计算c_t 必须先变换维度
        c_t = torch.bmm(a_t.permute(0,2,1),enc_rnn_output)

        return c_t
    
class Decoder(nn.Module):
    def __init__(self,input_dim,ebd_dim,hidden_dim,
                 dropout=0.2):
        #decoder解码就不用不上bidirectional
        super().__init__()
        self.ebd = nn.Embedding(input_dim,ebd_dim)
        self.rnn = nn.GRU(ebd_dim,hidden_dim*2,batch_first=True,
                          dropout=dropout)
        self.atten = Attention()
        #*4是因为原本decoder_rnn后是*2 
        # 在做贡献值矩阵c_t和decoder_rnn的拼接后再翻一倍
        self.atten_fc = nn.Linear(hidden_dim*4,hidden_dim)
        self.fc = nn.Linear(hidden_dim,input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input,state_c,enc_output):
        #无论有没有融入atention机制
        # 每次都把state_c新的长期记忆状态传入用于下一个token解码
        out = self.ebd(input)
        #无论encoder最后用拼接还是加和 得到的c都是2维
        #为保持和out形状一致 给state_c增加1个维度 才能运算 
        #相当于decoder中初次传入rnn的初始状态
        dec_output,h_t = self.rnn(out,state_c.unsqueeze(0))
        c_t = self.atten(enc_output,dec_output)
        #引入从encoder中挖掘到的隐含信息c_t到decoder解码
        cat_out = torch.cat((c_t,dec_output),dim = -1)
        #线性运算并引入非线性因素
        out = torch.tanh(self.atten_fc(cat_out))
        y_hat = self.fc(out)

        return y_hat,h_t #h_t就是解码下一个token要用的新state_c
        #这里返回的state_c又融合了新的记忆

class Seq2Seq(nn.Module):
    def __init__(self,enc_input_dim,dec_input_dim,ebd_dim,hidden_dim,
                 dropout = 0.2,num_layers= 2):
        super().__init__()
        self.encoder = Encoder(enc_input_dim,ebd_dim,hidden_dim,
                               dropout=dropout,num_layers=num_layers)
        self.decoder = Decoder(dec_input_dim,ebd_dim,hidden_dim,
                               dropout=dropout)

    def forward(self,enc_input,dec_input):
        state_c,enc_output = self.encoder(enc_input)
        dec_output,h_t = self.decoder(dec_input,state_c,enc_output)

        return dec_output,h_t
    
if __name__ == "__main__":
    #测试
     # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10


    seq2seq = Seq2Seq(
        enc_input_dim = input_dim,
        dec_input_dim =input_dim,
        ebd_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    logits,_ = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape)  # 应该是 [b