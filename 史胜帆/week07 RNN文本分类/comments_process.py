#douban_movies_comments.csv数据处理
import csv
import jieba
import matplotlib.pyplot as plt
import pickle

#下载数据
#提取评论和评分 同时对评论分词
#只取vote 0 5的评论
comments_votes = []
with open("douban_movies_comments.csv","r",encoding="utf-8") as f:
    reader = csv.DictReader(f,delimiter=',')
    for line in reader:
        comment = line['content']
        vote = int(line['votes'])
        if vote in [0,5]:# vote == 0 或 ==5
            comment = jieba.lcut(comment)
            comments_votes.append((comment,1 if vote > 0 else 0)) #转01二分类 1好评0差评
print(len(comments_votes)) # 3870
#画图观察评论长度分布 去除字数过少和字数过多的评论
comments_len = [len(comment) for comment,vote in comments_votes]
plt.hist(comments_len,bins=100)
plt.show()
plt.boxplot(comments_len)
plt.show()
#观察发现 保留长度在10~150字符之间的评论
comments_votes = [cv for cv in comments_votes if len(cv[0]) in range(10,151)]
print(len(comments_votes))

#用pickle把处理完成的数据用二进制保存
with open("./dbmv_comments.pkl",'wb') as f:
    pickle.dump(comments_votes,f)

