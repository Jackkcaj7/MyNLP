#transformer 大模型等应用的主导架构
#可以理解为一个高级的EncoderDecoder
#为进一步解决长距离依赖问题而提出
#BLEU 模型评估指标
#
#核心:self-attention + multi-head-attetion = scaled dot-product attention
#transformer 分为2大部分 encoder decoder 都有输入
#encoder 输入 padding过后的idx_seq_metric再ebd 再叠加positional encoding
    #常见的NLP模型都会用RNN进行 位置相关信息的训练和提取 但transformer中没有
    #位置相关信息即RNN的h_t 是从1位置到t位置的记忆的累加 
    #记忆i距离位置t越近 在累加记忆状态里所占权重越多 也就是记忆越清晰
    #那么越远 记忆越弱 有遗忘性
    #因此从1位置到t位置有记忆(权重)衰减
#那么为了解决这种RNN记忆衰减的问题 引入positional encoding 
#既是模拟RNN这种机制 也是解决RNN这一问题
#positional encoding实现原理是
#为1个token_seq中的所有token赋予一个位置索引 
# 所有奇数索引位赋予sin值 所有偶数索引位赋予cos值
#得到的是一个矩阵 (token_seq_len,ebd_dim)  其值是符合上面规则的sin cos值
#刚好形状和ebd_idx_seq_metric相同 可以做相加运算 相当于给输入数据做了位置加权
#实现了像RNN一样从1位置到t位置 权重越弱的效果
#因此 positional encoding的目的是
    #1对transformer没有使用RNN的补偿
    #2为下一步self-attention做准备
#ebd + positional encoding ->都做QKV  即QKV是同源的
#同源QKV先进行head_i的划分再经过不同的Linear层->真正意义上的具有可解释性的QKV
#其中做Linear前的QKV都分别在ebd_dim维度上被拆分为不同的组
#这样的目的是 让模型在不同的子空间中学习到相关信息
#这样的拆分 可以理解为 [batch,seq_len,ebd_dim] ebd_dim=200
#把这200大小的一组向量 以50一组拆成4个子空间 即4个head
#这就是multi-head multi-head(Q,K,V) = concat(head1....headn)W
#其中head_i = Attention(QW_Q,KW_K,VW_V)
#head_i就是1个拆分出的子空间
#head_i的大小必须是ebd_dim // num_head 整除的结果

#切分出1个head_i原理 
#切分前 原始维度shape(batch,seq_len,ebd_dim) 其中 ebd_dim==num_head*head_dim
#切分后 shape(batch*num_head,seq_len,head_dim)
#这样切分 既把一个head切分出来 维度上又合理 切分前后各维度相乘后得到的结果相等
#对于经过线性层之前的QKV都是如此拆分成head_i
#multi-head的过程始终是再数据维度拆分而不是数据

#scaled dot-product attention 
#运算原理和seq2seq中的attention计算完全一致
#   (softmax(Q@K))@V 得到的结果可以理解为从V中提取出的具有贡献作用的认知经验
#其中softmax(Q@K)可以理解为得到的是token_i和所有token的关系map
#其中scaled 是对数据做缩放 对于整体分布分散的数据集中 集中的分散 使数据分布合理 既不过于分散也不过于集中 
#这样做的目的是 让做softmax时提取到更多有价值的信息 相当于让有价值数据更活跃
#attention(Q,K,V) = softmax(QK/sqrt(ebd_dim)) @ V 其中sqrt(ebd_dim)即scale
#运算后的attention(batch,seq_len,ebd_dim)维度是和输入的QKV维度相同(认准这里的返回维度即可)
#这就是sdpa的attention的结果

#再把sdpa过后的多头结果重新合并 先(batch*seq_len,ebd_dim) 再(seq_len,batch,ebd_dim) 
#这是对sdpa后的结果的再加工 目的是为后续运算做准备即残差连接(add) + norm

#残差连接
# 把输入数据和运算后的输出数据进行叠加 对抗模型深度过长带来的数据和权重衰退问题
# 相当于positional encoding + ebd的作用
# 可以理解为强化一下数据 再往后传递

# 再把残差连接后的结果归一化(norm)
#调整数据分布到合理区间 消除运算后带来的协变量偏移
#这里是自然语言处理 因此在1个token维度做layer归一化合理

#在论文模型图里scaled dot_product atten是写在multi-head atten里面的

# 在实际应用中 有时把norm放在multi-head之前比论文里放head切分之后效果更好
#因此pytorch中的实现 有norm_first norm两种实现方式

#add&norm后的结果 经过feed-forward 即一个MLP普通神经网络
#再把feed_forward输入和输出结果再进行1次add&norm

#整个这样的过程 从positional encoding+ebd 到最后1次add&norm
# 相当于1个高级的神经元cell 进行N次运行这个cell 就是 N × encoder
#整体来看 这就像一个 复杂高级的NN
#而其中的1个encoder的过程之中
#每1小部分的运算又像1个个神经网络中的神经元 N个encoder连接起来又回归到了普通NN 上一个encoder中每小部分连接到当前encoder中的没小部分
#这是一个从简单到复杂再回归简单的过程

#mask mask矩阵
#形状和atten=Q@K结果完全相同 下三角元素全为0 上三角元素全为负无穷的矩阵
#softmax(atten)过后 根据softmax公式 上三角元素全为0即权重为0
#含义在于 给模型一个上三角全为0的token_map让它去用每行的前几个字符预测下一个字符 相当于seq2seq中decoder用前一个token预测下一个token
#目的是让模型不看到一句话的后半部分即不完整矩阵去猜 提高模型性能
#用torch.triu()保留上三角元素 其余0
#torch.tril() 保留下三角元素 其余0
#条件填充 .masked_fill(a==x,b) a==x表示matric a中所有等于x值的元素 b是被添元素

#encoder_output(memory) 做K也做V传入decoder decoder的输入经过masked_att做Q

#Pytorch中的transformer只含有N × encoder 和 N × decoder两个模块 其他自己实现

#在encoder中加mask是为了增加模型性能
# 在decoder中加mask是必须的 因为decoder解码预测就是要用前一个token预测后面token 是decoder解码原理所必须的

# 
