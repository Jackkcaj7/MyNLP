# attention is all your need

#attention 
#可以理解为 学习一种新事物过程中可以获得的所有认知和经验的有机总和
#目的 关注不同来源的关键内容 整合后作用于最终目标
# 三要素QKV
#Q query 请求 目标
#K key 根据目标从资源中获得的关键经验
#V value 为完成目标所收集的大量资源
#sum(key) = attention

#2个模型 Encoder&Decoder 
# 也被合称AutoEncoder 通过学习输入的表示(关键信息)来重建输出 目标是min输入输出差异(loss)
#input---encoding--->hidden_features---decoding--->output
# encoder 把资源中的关键信息抽象出来
# decoder 用抽象出来的认知和经验结合当前情景和理解转化出来
#CV领域应用 语义标注 图像风格迁移 图像降噪 数字人物
#NLP领域应用 seq2seq

#seq2seq 基于RNN实现的序列编码解码技术
#原理 用2个RNN结构 
# 1个 N vs 1 的RNN作encoder抽象出资源 1个 1 vs N 的RNN作decoder解码
# N1rnn--->c抽象概要--->1Nrnn
#用2个的原因：RNN族网络可以解决序列特征提取问题 即N vs 1问题和N vs N问题
# 但是当输入输出不对齐时 标准的RNN族无法直接应用 即 N vs M 问题
# 因此引入seq2seq解决

#N vs 1 可以理解为 N个请求项 1个目标项
