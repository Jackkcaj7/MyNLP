#使用fasttext做词向量训练
#数据集使用红楼梦 hlm_c.txt
#fasttext要求提供 用空格分词的文档文件

import fasttext
import jieba

# #文档分词 用空格符连接分词
# with open("hlm_c.txt","r",encoding="utf-8") as f:
#     lines = f.read()

# with open("hlm_c_phase.txt","w",encoding="utf-8") as f:
#     f.write(' '.join(jieba.lcut(lines)))

model = fasttext.train_unsupervised("hlm_c_phase.txt",model = "cbow")
# #模型保存
# model.save_model("hlm_cbow.bin")
# #模型加载
# model = fasttext.load_model("hlm_cbow.bin")

#查看训练后的文档词汇表
print(model.words,len(model.words))
#查看训练好的词向量
vector = model.get_word_vector("钩")
print(vector)
#查看近邻词
print(model.get_nearest_neighbors("宝玉",k=5))
#分析词间类比
print(model.get_analogies("宝玉","黛玉","宝钗"))

