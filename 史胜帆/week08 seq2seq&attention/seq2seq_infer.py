# seq2seq模型训练和预测推理的输入及运行方式是不同的

"""
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""
import torch
import pickle
from EncoderDecoderAttenModel_full import Seq2Seq

if __name__ == '__main__':
    # 加载训练好的模型和词典
    state_dict = torch.load('seq2seq_state.bin')
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )
    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dvoc_inv = {v:k for k,v in dvoc.items()}

    # 用户输入
    enc_input = "Hi"
    # enc_input = "What I'm about to say is strictly between you and me"
    # enc_input = "I used to go swimming in the sea when I was a child"
    enc_idx = torch.tensor([[evoc[tk] for tk in enc_input.split()]])

    print(enc_idx.shape)

    # 推理
    # 设置最大解码长度 防止模型无法预测到<EOS>导致无限循环解码下去
    max_dec_len = 50

    model.eval()
    with torch.no_grad():
        # 编码器
        hidden_state = model.encoder(enc_idx)
        # hidden_state, enc_outputs = model.encoder(enc_idx)  # attention

        # 解码器输入 shape [1,1] 只输入第一个字符 让模型长期记忆状态c开始背书

        #(1,1)符合模型输入形状
        dec_input = torch.tensor([[dvoc['BOS']]])

        # 循环decoder 开始解码循环预测
        dec_tokens = []
        while True:
            if len(dec_tokens) >= max_dec_len:
                break #大于最大解码长度 不再解码预测字符
            # 解码器 
            # logits: [1,1,dec_voc_size]
            logits,hidden_state = model.decoder(dec_input, hidden_state)
            # logits,hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)
            
            # 预测出来的下个token 的index
            next_token = torch.argmax(logits, dim=-1)
            # 用反向索引词典返回index对应的token
            #next_token (1,1)降维(1)只要一个字符
            if dvoc_inv[next_token.squeeze().item()] == 'EOS':
                break
            # 收集每次token_index 【解码token序列   集合】
            dec_tokens.append(next_token.squeeze().item())
            # decoder的下一个输入 = token_index
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1) #防止维度不对齐

    # 输出解码结果
    print(''.join([dvoc_inv[tk] for tk in dec_tokens]))

    #发现结果并不好 即使输入是hi仍然翻译错误
    #随着输入句子长度增加 基本encoder decoder的性能大幅下降
    #原因是无法进一步消除长距离依赖的影响
    #对策 把attention机制引入seq2seq
    #利用encoder构建attention矩阵 获取更多信息 即把encoder的outputs引入decoder
    #再将这样的attention作用于decoder每一个用上一个token预测下一个token的过程中
    #修改EncoderDecoderModel 加入attention机制
    #如何引入 把encoder的outputs和decoder每个时间步的输出做运算 
    # 得到attention权重矩阵
    #1定QKV
    #2做QKV运算获取Attention
    #3attention与原Q做拼接 二者作用
    #4维度变换 非线性等操作输出新结果
    #attention运作的本质是提取某二者关系 把关系转换为权重 再抽取隐含信息