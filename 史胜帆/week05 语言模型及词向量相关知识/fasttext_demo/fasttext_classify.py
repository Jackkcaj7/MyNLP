#用fasttext做文本分类
#数据集用cooking_stackexchange.txt

import fasttext 

model = fasttext.train_supervised('cooking.stackexchange.txt',
                                  epoch = 10,dim = 100)
#预测效果与分类模型训练参数相关
print(model.predict("How much does potato starch affect a cheese sauce recipe?"))