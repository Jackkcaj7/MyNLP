#根据处理过的comments_fixed.txt 和加载的停用词表stopwards.txt 进行图书推荐

import csv #读取格式化的文件
import jieba # 文本分词
from sklearn.feature_extraction.text import TfidfVectorizer #tfidf
from sklearn.metrics.pairwise import cosine_similarity #余弦相似度
import numpy as np

#1收集数据
def collect_data(file):
    #初始化
    book_comments = {} # 字典存储键值对 {书名:评论1分词+评论2分词+评论3分词....} 
    #把同一本书的所有评论看作一个大字符串 因此拼接起来

    with open(file,"r",encoding="utf-8") as f:
        #csv.DictReader() 可以识别标题列 可以通过属性名检索出对应的属性列
        reader = csv.DictReader(f,delimiter='\t')  
        #delimeter参数识别不同字段的分隔符 默认,

        #收集所有书名和对应评论
        for item in reader:
            book = item['book']
            comment = item['body']
            
            if book == '':  # 防止书名为空
                continue
            if comment == None:
                continue
            comment_words = jieba.lcut(comment) #使用jieba分词对文本分词
            #每拿到一本新书 创建一个值的列表[]
            book_comments[book] = book_comments.get(book,[])
            #字典get()方法 如果key对应的值存在返回值 如果=None 返回指定内容 此处是[]
            book_comments[book].extend(comment_words) #list合并extend
    
    return book_comments

# 测试    
if __name__ == "__main__":
    #2计算重要性和相似度
#词汇重要性指标采用TF-IDF  文本相似性采用余弦相似度
#sklearn.feature_extraction.text.TfidfVectorizer
#sklearn.metrics.pairwise.cosine_similarity

    #加载停用词列表
    stop_words = [line.strip() for line in open("stopwords.txt","r",encoding="utf-8")]
    #加载book_comments数据
    book_comments = collect_data("comments_fixed.txt")
    print(len(book_comments))
    #构建TFIDF矩阵 行是所构建字典的key数(不同书) 列是整数据集合的不同分词数
    #一行代表一本书对应的评论 是一个词向量 
    # 该书评论中出现的分词对应这一行向量的元素有值 没出现的无值
    vectorizer = TfidfVectorizer(stop_words = stop_words)
    #整个数据集合的分词构成一个词向量
    #分词的结果每个分词会带''号 要生成tfidf矩阵 要先去掉''把每个元素用空格连接
    #fit_transform 填充+变换
    tfidf_metric = vectorizer.fit_transform([' '.join(comments) for comments in book_comments.values()])
    
    print(tfidf_metric.shape) #shape(233, 75209)

    #计算图书和图书两两之间的余弦相似度
    similarity_metric = cosine_similarity(tfidf_metric) 
    #shape(233,233) (i,i) = 1  是一个实对称矩阵

    #3推荐环节
    #给出所有书籍名称
    books = list(book_comments.keys())
    print(books)
    #输入图书名称
    book_name = input("请输入图书名称：")
    
    #要把相似度矩阵的元素对应到具体哪本书的名称 
    # 就要用数组索引把相似度数值和书名关联起来
    #调用list的.index()直接用值获取位置索引
    book_idx = books.index(book_name)
    #argsort升序排列元素 取反再升序 容易正向取值
    #取最推荐的前5本书的索引 0索引是自身与自身的相似度 必为1最大 所以从1索引取
    recommd_books_idx = np.argsort(-similarity_metric[book_idx])[1:6] 
    #用索引输出书名
    for idx in recommd_books_idx:
        print(f"《{books[idx]}》\t 相似度：{similarity_metric[idx][book_idx]}")
    

        
