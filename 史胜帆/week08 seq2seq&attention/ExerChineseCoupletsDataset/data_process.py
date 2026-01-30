#couplets.txt数据集预处理
#因为是对联 要求上下联字数相同且有对偶性 所以逐字分词为好 其他按词汇分词不好

import torch
import json
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

#读取数据
def get_data(file_name_in,file_name_out):
    enc_data,dec_data = [],[]
    #数据导入
    data_in = open(file_name_in,'r',encoding='utf-8')
    data_out = open(file_name_out,'r',encoding='utf-8')
    #分词和保存
    #对联逐词分词合理
    for in_,out_ in zip(data_in,data_out):
        enc_tks = in_.strip().split()
        dec_tks = out_.strip().split()
        enc_data.append(enc_tks)
        dec_data.append(dec_tks)
    
    #检验上下联个数是否一致
    assert len(enc_data)==len(dec_data) ,"编码数据解码数据长度不一致"

    return enc_data,dec_data

#构建词表 词典索引
class Vocabulary:
    def __init__(self,vocab_idx):
        self.vocab_idx = vocab_idx

    @classmethod
    def get_vocab(cls,enc_data,dec_data):
        vocab = set()
        for enc,dec in zip(enc_data,dec_data):
            vocab.update(enc)
            vocab.update(dec)
        vocab = ['<pad>']+['<unk>']+['<s>']+['</s>']+list(vocab)
        #词典索引
        vocab_idx = {tk:i for i,tk in enumerate(vocab)}
        return cls(vocab_idx) #返回的时类

#闭包 X y的文本矩阵转带<s></s>索引的tensor序列矩阵 padding
def get_proc(vocab_idx):
    
    def batch_proc(data):#batch数据传到这里
        #tensor_ids_sequence_metric
        enc_ids,dec_ids,labels = [],[],[]
        for enc,dec in data:
            enc_idx = [vocab_idx['<s>']] + [vocab_idx[tk] for tk in enc] + [vocab_idx['</s>']]
            dec_idx = [vocab_idx['<s>']] + [vocab_idx[tk] for tk in dec] + [vocab_idx['</s>']]
            enc_ids.append(torch.tensor(enc_idx))
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            labels.append(torch.tensor(dec_idx[1:]))
        #padding (batch,max_token_seq_len) 默认用0索引对应的token填充
        enc_input = pad_sequence(enc_ids,batch_first=True)
        dec_input = pad_sequence(dec_ids,batch_first=True)
        targets = pad_sequence(labels,batch_first=True)

        return enc_input,dec_input,targets

    return batch_proc

if __name__ == "__main__":
    enc_data,dec_data = get_data("ExerChineseCoupletsDataset/fixed_couplets_in.txt",
                                 "ExerChineseCoupletsDataset/fixed_couplets_out.txt")
    vocab_idx = Vocabulary.get_vocab(enc_data,dec_data)
    vocab_idx = vocab_idx.vocab_idx #这才是vocab_idx 而不是cls(vocab_idx)

    #构建符合dataloader输入要求的元组数据集
    dataset = list(zip(enc_data,dec_data))
    #batch切分 回调(文本矩阵转序列矩阵 padding补齐 )
    datald = DataLoader(dataset,batch_size = 20,shuffle = True,
                        collate_fn = get_proc(vocab_idx))

    #数据缓存
    print(len(enc_data),len(vocab_idx))
    #后面要用 要存的是词典表索引 不是词表vocab
    with open("ExerChineseCoupletsDataset/vocab_idx.bin",'wb') as f:
        pickle.dump(vocab_idx,f)
    with open("ExerChineseCoupletsDataset/encoder.json","w",encoding='utf-8') as f:
        json.dump(enc_data,f)
    with open("ExerChineseCoupletsDataset/decoder.json",'w',encoding='utf-8') as f:
        json.dump(dec_data,f)



