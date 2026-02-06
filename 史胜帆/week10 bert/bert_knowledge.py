#bert 双向Transformer编码器表征
#一种预训练语言模型(结构)
#LSTM ELMo->openAI GPT->google BERT
#本质 transformer中的encoder block
#输入和输出的长度是完全对齐的
#保留原有
#创新点在于 模型训练的方法 
    #1 位置编码positional encoding硬编码替换为随机初始化的可训练参数position embedding
    #即位置编码换成训练参数
    #2 给每个token_seq添加[cls]标签作为类别用于文本分类 用[sep]分隔标签统一标注起始结束位置
    #这样用[sep]连接2句话当1句话加入encoder 可用于2句话的上下文关联推理
    #3 基于句子的segment embedding 输入的文本以[sep]分隔为不同句子
    #一个句子i中的每个token具有相同值和大小的向量
    #有几个句子就有几种不同的segment embedding_i
    # 2、3归纳起来就是 通过引入特定token添加句向量参数
    #4子词切分 playing ->play + ##ing
    #因为一个词通过构词可以形成多个词汇 工程量庞大 因此找出构词规律做子词拆分
    #适用于拼写式语言 好处在于 减少工程量还可以发掘变形词含义
    #5最终把 词向量+句向量+位置向量 三者加起来组成一个向量
#但是只通过这样的输入去进行token对齐的模型输出 训练出的模型没有泛化性
#对策 进行2个预训练任务 1masked language model 2next sentence predict
#1MLM mask遮盖 按比率随机遮盖或替换token_seq中的非[cls][sep]token
#规则 在token_seq中选择15%的词 
# 在这15%中的80%用[mask] 10%用原始token 10%用随机token
#那么也就是说在MLM预训练任务中 输入一直在随机换token在变 要输出的答案是不变的
#起到了训练效果
#那么有这样的效果主要是self-attention起了作用 分析到了句子中每个词之间的关联
#因此bert特别适合做填充任务 不适合做生成任务
#由于在MLM训练过程中找到了每个token之间的关联关系 也训练出了[cls]直接可以分类
#因此做文本分类效果也比较好
#2NSP 用上一句预测下一句 效果不好 作用不大 因此现在使用NSP加不加句向量都没关系

#预训练模型(已经被官方训练好 不用再手搓训练)调用
#from transformer import pipeline
#分解pipeline观察pipeline运行原理 为自己搭建一个pipeline做准备
#Pipeline的运行过程包含3步：预处理(tokenizer) 模型运算 后处理(像decoder后MLP)

#先解决第2步 加载模型
#用transformer中的Auto系列去通过name或path去加载本地或网站中预训练模型
#bert模型的 input参数: token_type_id token_idx masked_token
# bert output:shape[batch,token_seq,hidden_dim] ->pool [batch,hidden_dim] 
#pool把一个token_seq即一句话压缩成一个hidden 这样的结果用于做文本分类
#bert模型返回1个字典包含2个结果 last_hidden_state pooler_output

#Autoconfig 加载模型配置信息 里面是模型资源的参数、词表大小等信息 
#可以用这些配置信息创建bert模型 但创建出来的是未经训练的
#config也是模型自身的一个属性 可以调用查看

#AutoTokenizer分词器 
#作用 1原始文本拆分为tokens 2加载词表vocab
# 3把token_seq 转换为tensor_token_idx_seq
#4 tokenizer.encode 和.decode实现text token_idx_seq互转 
#5tokenizer的最佳使用方式 tokenizer([seq1,seq2....seqn])
#返回的就是模型训练要用的数据

#post_processing部分
#这一部分 hf里给提供有特定任务的出口模型 
# 如果没有提前声明这一网络层的参数 模型会自动先生成 但是是未经训练的 
# 所以后期还要自己训练微调

#因此hf中的预训练模型 不是用来预训练的 是官方已经预先训练好的模型 
#也就是说 预训练模型是用来调用在根据自己的具体需求定制去做微调的
# AutoModel ->AutoTokenizer ->AutoFor具体post_pocessing任务
#因此 预训练模型的使用流程
#1加载预训练模型AutoModel 定制输出端任务(也可加载预训练模型中已有的输出任务模型)
#2数据预处理 停用词 非法字符等
#3dataloader collate_fn里使用AutoTokenizer将文本数据转换成模型输入数据
#4搭建模型 损失函数 优化器
#5训练模型
#6观察损失 调参迭代
#7模型保存