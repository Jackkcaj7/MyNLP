#用训练好的参数和模型及数据进行推理
#1加载模型 参数 词典表索引
#2输入待预测文本序列txt 转 索引序列
#3输入索引序列到encoder获取state_c
#4准备token <s> shape(1,1)作为decoder第1个输入
#5循环decoder 让他不断输出预测token 背书
#

import torch
from EncoderDecoderAttenModel import Seq2Seq
import json
import pickle
from data_process import get_data,Vocabulary,get_proc
import random

if __name__ == "__main__":
    #数据加载

    with open("ExerChineseCoupletsDataset/encoder.json",'r',encoding='utf-8') as f:
        enc_data = json.load(f)
    with open("ExerChineseCoupletsDataset/decoder.json",'r',encoding='utf-8') as f:
        dec_data = json.load(f)
    with open('ExerChineseCoupletsDataset/vocab_idx.bin','rb') as f:
        vocab_idx = pickle.load(f)

    state_dict = torch.load('seq2seq_state.bin')
    
    #模型搭建
    model = Seq2Seq(enc_input_dim=len(vocab_idx),
                    dec_input_dim=len(vocab_idx),
                    ebd_dim = 256,
                    hidden_dim=360,
                    dropout=0.2,num_layers = 2)
    model.load_state_dict(state_dict)

    #创建反向索引词典 用于返回解码token
    idx_vocab = {idx:tk for tk,idx in vocab_idx.items()}

    #随机选取样本
    rd_idx = random.randint(0,len(enc_data))
    enc_input = enc_data[rd_idx]
    dec_output = dec_data[rd_idx]
    enc_idx_seq = torch.tensor([vocab_idx[tk] for tk in enc_input])

    #开始推理预测
    #设置最大解码长度 防止无限循环
    max_dec_len = len(enc_input)

    model.eval()
    with torch.no_grad():
        #直接调用大类里面的小类 Seq2Seq里的Encoder
        state_c,enc_output = model.encoder(enc_idx_seq)
        #dec_input <s> (1,1)
        dec_input = torch.tensor([[vocab_idx['<s>']]])

        #循环decoder 解码背书
        dec_tokens = []
        while True:
            if len(dec_tokens) >= max_dec_len:
                break

            dec_ouput,state_c = model.decoder(dec_input,state_c,enc_output)

            #预测出来的token索引 选词表中概率最大的
            token_pred_idx = torch.argmax(dec_ouput,dim = -1) #(1,1,1)
            token_pred = idx_vocab[token_pred_idx.squeeze().item()]
            #预测到</s> 终止预测
            if token_pred == '</s>':
                break
            #收集token
            dec_tokens.append(token_pred)
            #预测出来的token作为下一循环decoder的输入
            dec_input = token_pred_idx.squeeze().item()
            #因为encoder的state_c输出是2层作用的结果 decoder只能1层 所以view
            state_c = state_c.view(1,-1)

        #输出解码token序列
        print(f"上联：{''.join(enc_input)}")
        print(f"下联：{''.join(dec_output)}")
        print(f"预测出的下联：{''.join(dec_tokens)}")



