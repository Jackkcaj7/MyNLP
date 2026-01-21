#用SentencePiece分词工具对红楼梦hlm_c.txt文件 分词

import sentencepiece as sp

#引入外部文件训练分词模型
#sp.SentencePieceTrainer.Train(input = "hlm_c.txt",model_prefix = "hlm_mod",vocab_size = 10000)
#model_prefix 是训练出来模型的名称的前缀
#得到2个文件 hlm_mod.model hlm_mod.vacab

#加载模型进行分词
spp = sp.SentencePieceProcessor(model_file = "hlm_mod.model")
#用模型分词
print(spp.EncodeAsPieces("道可道，非常道；名可名，非常名。"))
