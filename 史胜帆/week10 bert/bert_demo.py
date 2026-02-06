from transformers import pipeline

classifer = pipeline(
    "text-classification",
    model = "uer/roberta-base-finetuned-dianping-chinese"
) #参数任务类型 模型名称 如果模型在kaggle中不存在 会下载到本地
output = classifer(["~"])
print(output)

#用tansformers中的Auto类去加载预训练模型
from transformers import AutoModel

bert = AutoModel.from_pretrained("google-bert/bert-base-chinese")
print(bert)

"""
 (word_embeddings): Embedding(21128, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
模型输入embedding和论文里一致

模型主体是12×encoder
""""""```

#调用bert
#transformer默认加载出来的模型是pytorch类型 不用转换模型类型
import torch
#构造bert input
token_idx = torch.randint(1,100,(4,10)) #(batch,token_idx_seq)
#模型调用
res = bert(token_idx)
#print(res)
last_hidden_state,pooler_output = bert['last_hidden_state'],bert('pooler_output')

#加载模型配置信息
from transformers import AutoConfig,BertModel

config = AutoConfig.from_pretrained("google-bert/bert-base-chinese")
#print(config)
#通过congfig创建bert模型(未训练)
new_model = BertModel(config) 
#print(new_model)
#config是模型自身属性 可以调用查看
print(bert.config)


#tokenizer 分词器
from transformers import AutoTokenizer
#加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

#tokenizer的应用
#1文本分词
text = "HF Mirror x 趋动云，即刻体验从模型、数据、项目的发现到部署、运行的完整流程"
tokens = tokenizer.tokenize(text)
#分词token_seq转序列索引
#tokenizer.encode文本转索引
token_idx = tokenizer.encode(text)
print(token_idx)
raw_text = tokenizer.decode(token_idx)
print(raw_text)

#tokenizer的最佳调用方式
text = ["政教及大企业","尊敬的联想用户：基于个人信息保护领域法律","我们使用这些信息来满足法律或法规要求；维护我们网站的安全性和完整性"]
model_inputs = tokenizer(
    text,
    return_tensors = "pt", #返回pytorch类型数据
    padding = True,# padding填充 以最长seq长度为基准
    truncation = True #超过bert embedding最大长度512就裁剪掉超的
)

#调用生成的模型输入数据 model_inputs 放进模型中运算 的到运算结果
bert(**model_inputs) # 其中**表示自动将字典中key和方法参数名对应 然后传入其value
#还有一种方法就是像普通方法调用一样直接写
#token_type_ids是句向量参数 可以在输入文本都是句话的前提下不加这个参数

#定制模型输出 即post-processing部分
from transformers import AutoModelForSequenceClassification
post_model1 = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-chinese",
                                                                num_labels = 5)
print(post_model1) #如果没有提前声明这一网络层的参数 模型会自动先生成 
#也就是说 这一部分 hf里给提供有特定任务的出口模型 但是是未经训练的 所以后期还要自己训练微调

