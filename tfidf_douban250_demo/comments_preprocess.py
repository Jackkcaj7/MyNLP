# 获取doubanbooktop250comments数据集https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/X20PS1
#1下载到本地
#观察数据集 发现数据集不规整 有些评论出现在下一行
# 有6种属性

#2创建一个新文件 存储修复后的文件
fixed = open("comments_fixed.txt","w",encoding="utf-8")
#读取原文件 读取每一行直接放进一个列表里
lines = [line for line in open("doubanbook_top250_comments.txt","r",encoding="utf-8")]
# 提取要用的信息 书名book 评论文本body
# 发现数据集中每个样本不同属性之间用'/t'制表符连接
# 因此可以把读取出来的每一行用制表符切分出来收集

#初始化
pre_line = ""
for i,line in enumerate(lines): #使用可返回带位置索引的内容的迭代器
    #保存标题列 直接把标题列写进fixed文件
    
    if i == 0:
        fixed.write(line)
        continue
    #获取去掉制表符的每行
    terms = line.split("\t") #得到的是list
    
    # 只能每次合并上一行 合并的不能是当前行
    if len(terms) == 6:
        pre_line.split("\t")[0] == terms[0]
        fixed.write(pre_line.strip() + "\n")
        pre_line = line.strip() #去除字符串两侧空白字符
    else:
        pre_line += line.strip() #字符串合并+ 

# 3 文本分析 要先分词 分词过程中文本中有一些像‘的 地 得’这种 没有分析价值的词汇要过滤掉 这种词汇成为停用词
# 过滤无效词用停用词表 stopwards.txt
# 在TfidfVectorizer()矩阵时 传入stopwords 做无效词筛选