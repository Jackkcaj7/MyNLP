#利用NER模型实现动态学习率 混合精度 DDP训练实现

import os 
import numpy as np
from transformers import AutoModelForTokenClassification,AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments,Trainer
import torch
import evaluate,seqeval
from datasets import load_dataset

def train(local_rank):
    #加载数据集
    ds = load_dataset('ds_msra_ner')
    #构建标签列表的实体索引字典
    entities = ['0'] + list({'PER','LOC','ORG'})
    tags = ['0']
    for ent in entities:
        tags.append('B-'+ent.upper())
        tags.append('I-'+ent.upper())
    
    entity_index = {entity:i for i,entity in enumerate(entities)}

    #基于bert预训练模型的分词器 实例化 后续可直接生成tokenize后的模型输入参数
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    def data_input_proc(item):
        #此处item是基于batch的
        #先把item的text拆分为字符 再用tokenizer转为符合bert格式的输入数据 是基于token_seq的
        batch_texts = [list(text) for text in item['tokens']]
        #text已经被拆分为word字符 因此input_data要设置is_split_words参数
        #使用tokenizer将字符序列文本转换为bert格式的输入数据
        #tokenizer的结果是1个字典
        #  (1,text_num,token_seq)
        input_data = tokenizer(batch_texts,truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False,#没有[cls][sep]
                                max_length = 256,padding='max_length')
        #往input_data里添加labels数据
        input_data['labels'] = [lbl[:256] for lbl in item['ner_tags']]
        
        return input_data

    #把模型输入数据映射到原始数据集中去
    ds2 = ds.map(data_input_proc,batched=True) #基于batch的 所以item也是
    #构建索引标签字典和标签索引字典
    id2label = {id:tag for id,tag in enumerate(tags)}
    lable2id = {tag:id for id,tag in enumerate(tags)}

    #搭建模型
    model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-chinese',
        num_lables = len(tags),
        id2label = id2label,
        lable2id = lable2id
    )
    #模型分配到不同进程GPU
    model.to(local_rank)

    #设置参数声明 用于Trainer
    args = TrainingArguments(
        output_dir = 'ner_train', #模型训练工作目录 保存的存盘文件
        num_train_epochs = 3, #模型训练次数
        save_safetensors=False, #后续用torch.load加载模型
        per_device_train_batch_size =16,#训练批次
        per_device_eval_batch_size = 16,#评估批次
        report_to = 'tensorboard', #只把训练记录输出到tensorboard
        eval_strategy = 'epoch', #评估策略
        local_rank = local_rank, #当前进程
        fp16 = True, #使用混合精度
        lr_scheduler_type = 'linear', #动态学习率
        warmup_steps = 100, #动态学习率达到指定学习率的预热步数
        ddp_find_unused_parameters = True, #优化DDP性能 同步有梯度参数 忽略无梯度参数

    )

    #定义模型评估函数 主要目的是把模型输出结果去除pad元素对齐真是标签
    #再封装seqeval的评估计算
    def compute_metric(output):
        #output是1个元组 (模型预测值，模型真实值序列)
        #其中 模型预测值tensor(batch(num_sentence),num_token_seq,num_tags)
        #模型真实值tensor(batch(num_sentence),num_token_seq)
        #其中 模型真实值中的num_token_seq是1各token对应1个tag
        
        #获取评估对象
        seqeval = evaluate.load('seqeval')
        #加载输出值和标签
        pred,lables = output
        #对齐输出值和标签
        pred = torch.argmax(pred,dim = 2) #相当取1个分类任务的最大值

        #转换评估数据为seqeval所要求的输入数据
        #seqeval的输入数据要求是(模型预测值的标签序列，模型真实值标签序列)
        # -100是DataCollatorForTokenClassification中pad的默认值
        pred =[[tags[p] for p,l in zip(ps,ls) if p != -100] for ps,ls in zip(pred,lables)] 
        lables = pred =[[tags[l] for p,l in zip(ps,ls) if l != -100] for ps,ls in zip(pred,lables)] 
        #用seqeval评估
        res = seqeval.compute(predictions=pred,references = lables)

        return res
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,padding =True)

    #构建Trainer开始训练
    trainer = Trainer(
        model,
        args,
        train_dataset=ds2['train'],
        eval_dataset=ds2['test'],
        data_collator=data_collator,
        compute_metrics = compute_metric
    )

    #开始训练 
    trainer.train()

#定义主函数 目的是用torchrun来实现分布式训练 不用手动配置分布式环境变量
#功能是 解析命令行local_rank参数 传入trainer 为DDP传入核心设备编号参数 
#bash运行 torchrun --nproc-per-node = k Distributed_ner_train.py
#其中k是GPU数量
def main():
    import argparse #python内置的命令行参数解析模块 代替手动操作sys.argv
    parser = argparse.ArgumentParser() #创建参数解析器对象
    #添加local_rank参数：
    #  --local_rank：可选参数（分布式训练框架如torchrun会自动传入）
    #  type=int：强制转为整数（GPU编号是整型）
    #  default=0：未传参时默认用0号GPU（兼容单卡场景）
    parser.add_argument('--local_rank',type=int,default = 0)
    #解析命令行参数（比如运行时传入 --local_rank 1 则args.local_rank=1）
    args = parser.parse_args()
    #将local_rank传入train函数（用于指定训练的GPU设备）
    train(args.local_rank)

if __name__ == '__main__':
    main() 
