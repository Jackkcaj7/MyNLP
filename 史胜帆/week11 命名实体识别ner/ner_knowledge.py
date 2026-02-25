#命名实体识别 NER
#在自然语言中抽取感兴趣的实体名词
#这里实体是广泛的 可以是名词 动词等
#可以理解为 对自然语言中'有意义的内容'打上标签 划分到类别

#标注方式有 序列标注 指针标注 全局指针标注
#序列标注BIO三位标注法 begin inside outside(不属于任何类型) 
    #   只能解决非嵌套文本 比如 中国北京天安门 其中北京天安门也是一个实体

#指针标注 即范围标注 只标注每个实体的起始结束位置 对嵌套实体标注不好
#全局指针标注 token_map 行表示起始位置 列表示结束位置 嵌套非嵌套都可以
    #全局指针标注结果是一个上三角矩阵 因为句子不可能倒着读

#基于bert的NER
#1datasets 如果人工去构建可训练的数据集 任务量大 繁琐
    #用transformers框架提供的 AutoModelForTokenClassification 自带loss_fn
    #快速构建基于bert的NER模型
#因此 使用hf中官方提供的数据集库datasets 方便加载 为模型快速准备数据
#CLUE中文语言理解测评标准   对比于外国的BLUE
#在hf中datasets中选择CLUENER数据集 用于模型训练

#由于在基于中文预训练bert模型的词典中 数字字符串和英文单词不会按字符拆分
#比如 2000年1月add 之拆分为 2000 年 1 月 add
#所以出现字符和token不对齐的情况 比如 2000和 2 0 0 0
#这一情况在训练过程中会影响ner的准确性
#调整要求 1个字符对应1个标签 而不是2000这1个token对应一个实体标签
#对策  1手动算法对齐标注 
    # 2tokenizer时设置return_offset_mapping得到word和token的联系再手动调
#tokenizer返回结果的.word_ids(i)
# 返回batch中第i个token_seq对应原始文本以字符拆分而不是token的索引位置序列



