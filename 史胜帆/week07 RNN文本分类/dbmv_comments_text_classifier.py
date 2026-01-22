#用处理完成的dbmv_comments.pkl数据做文本分类
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 
#对索引序列张量做pad填充 1个batch内填充长度不同的张量到长度相同

#加载语料dbmv_comments.pkl
with open("dbmv_comments.pkl",'rb') as f:
    comments_votes = pickle.load(f)

# #构建词汇表
# vocab = set()
# for cv in comments_votes:
#     vocab.update(cv[0])
# print(len(vocab)) #12279

# #构建词典表索引
# vcb_idx = {vcb:idx for idx,vcb in enumerate(vocab)}
# print(vcb_idx)

# #词向量embedding(词嵌入) 索引->向量
# #那么 一个词向量就是n维空间中的一个点 其中n是向量维度 这样的向量具有计算性质
# #相当于就是带入一组embedding参数来代表输入的词汇 参与模型训练 成为最终模型的一部分
# ebd = nn.Embedding(len(vocab),100) #(len(vocab),features) 

# #待处理文本 转 索引序列
# comments_idx = []
# for cv in comments_votes:
#     idx_sery = [vcb_idx[word] for word in cv[0]]
#     comments_idx.append(torch.tensor(idx_sery))

#索引序列 转 向量矩阵




#用类方法写构建词表 并构建词典表索引
class Vocabulary:
    def __init__(self,vocab):
        self.vocab = vocab
    
    @classmethod
    def build_from_doc(cls,doc):
        vocab = set()
        for line in doc:
            vocab.update(line[0])
        #添加 UNK PAD
        vocab = ['PAD','UNK'] + list(vocab)
        #构建词典表索引
        vcb_idx = {vcb:idx for idx,vcb in enumerate(vocab)}

        return cls(vcb_idx)

vcb_idx = Vocabulary.build_from_doc(comments_votes) #对象vcb_idx

ebd = nn.Embedding(len(vcb_idx.vocab),100)

#用回调函数在dataloader里做文本转索引序列 
def txt_convert_ebd(batch_data):
    comments,votes = [],[]
    for cmt,vt in batch_data:
        comments.append(torch.tensor([vcb_idx.vocab.get(word,vcb_idx.vocab['UNK']) for word in cmt]))
        votes.append(vt)
    #PAD填充 用pad_sequence做 输入是UNK过后的索引序列矩阵张量
    idx_seq = pad_sequence(comments,batch_first=True,padding_value=vcb_idx.vocab['PAD'])
    labels = torch.tensor(votes)

    return idx_seq,labels
#batch切分 + 自定义方法加工整理
#自定义方法对batch打包好的数据再加工整理 再传给模型训练 collate_fn
#collate_fn 回调函数 被动被系统调用 这里只会在batch数据被加载后才会调用
cvs_dl = DataLoader(comments_votes,batch_size=32,shuffle=True,
                    collate_fn=txt_convert_ebd)


#构建分类模型
class MyModel(nn.Modeule):
    def __init__(self,vocab_size,ebd_dim,hidden_size,num_classes):
        super().__init__()
        self.ebd = nn.Embedding(vocab_size,ebd_dim,padding_idx=0)
        # 指定padding_idx的索引值 避免索引0对应的向量参与训练 
        self.rnn = nn.RNN(ebd_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,input):
        #input shape(batch,num_seq) num_seq是1个batch中索引序列的个数
        ebd = self.ebd(input)
        output,h_t = self.rnn(ebd)
        y_hat = self.fc(output[:,-1,:])

        return y_hat
    
model = MyModel(len(vcb_idx.vocab),100,128,2)
