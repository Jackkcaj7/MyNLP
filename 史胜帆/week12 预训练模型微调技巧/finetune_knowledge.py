# 预训练模型微调技巧  治大国如烹小鲜

#固定学习率 
#问题 模型梯度达到一个稳定期 损失函数很难进一步优化 出现局部最优点和鞍点

#对策 
#1微调学习率
#动态循环学习率 让学习率在合理区间内循环变化
#原理 从训练开始0缓慢增长到指定大小 待模型参数初步稳定后再采用正常策略调整lr
#优点 具有比动量更强的跳过局部最优点找到全局最优点的能力 且收敛速度更快

#微调学习率的应用 warmup  
#是transform模型训练的关键步骤 没有它可能无法收敛或者收敛到糟糕情况
#不在于少年得志 在于慢慢成长 防止模型过度自信
#原理 从训练开始0缓慢增长到指定大小 待模型参数初步稳定后再采用正常策略调整lr
    # 而不是一开始就用指定大小的固定lru训练
#不warmup的危害 模型一开始就快速学习 由于梯度消失和后层网络敏感
    #恶性循环 前层没学好 后层学的快 模型过度自信 达到局部最优点不再下降
#目的 给模型足够时间预热 抑制后层学的过快 促进每层同步优化

#差分学习 模型每层使用不同的学习率

#通用语言模型微调 ULMFiT  预训练+微调
#创新点 1差异化微调(差分学习率) 2斜三角学习率 3渐进解冻 从最后一层逐步解锁训练 防止一次性调整所有层导致遗忘
#创新点的目的 防止学了新的忘了旧的
#ULMFiT的意义 让模型更通用 让NLP像人类一样 先学通用语言能力
    # 再用少量任务数据快速适应 针对特定任务 这尤其适合数据量少的场景

#Transformers中的学习率微调
#transformers.AdamW 基于权重衰减的Adam
#动态学习率 提供scheduler来实现学习率的变更

#要在模型训练上设置上面的动态学习率 要进行transformers动态学习率规划
#将named_parameters()的参数分组成预训练参数和下游任务参数 再分别设置参数数组
#学习率调度器设置scheduler 其中train_steps是模型训练调用batch_size的总次数
#train_steps=num_datasets // batch_size * epochs
#使用 scheduler.step()

#2自动混合精度  必不可少 加速非常有效 至少加速为原来的4倍
#在训练中将单精度FP32和版精度FP16格式相结合 比如把2个FP16放进1个FP32里去运算
#提高模型运行效率和精度
#torch.amp.GradScaler() 模型梯度运算的缩放器 更快梯度计算 算1个->算2个


#3分布式多卡训练
#并行训练 将任务分配到多个设备上并行执行 加速处理速度 但每个设备上的参数都一样
#关键节点在于 如何保证每个设备上的参数都一样
#3中并行处理方式 DataParallel(DP) DistributedDataParallel(DDP)
#3.1 DataParallel 
#原理 
    # 先把模型整个训练过程分为3个阶段 forward loss backward
    #在每一个阶段都会把训练任务分发到不同的GPU中去(副本) 分发顺序是 数据data 模型model(包括参数 所以被分发任务的GPU的参数都是完全相同的 差异点在于每个GPU接收到的数据样本是不同的)
    #那么同一个模型经历不同训练样本输出不同训练结果
    #GPU分发数量 假设有i个GPU 那么任务会被分发到i个GPU中去 其中第1个GPU既是任务分发者又是任务接收者
    #forward阶段 分发数据 分发模型 前向运算
    #loss阶段 分发和样本数据对应的标签数据到对应的GPU 损失计算 每个GPU得到不同损失
    #backward阶段 每个GPU各自计算梯度 每个GPU再各自对所接收到的模型部分反向传播梯度 把不同GPU上的不同模型部分梯度结果汇总到一起存入GPU1
    #使用 model = model.DataParallel(model) model.to(device)
    #弊端 模型分发和梯度汇总 带宽消耗大 因此DP的优点也是弊端

#3.2 DistributedDataParallel
#原理 
    #并行训练的每个GPU_i 都使用相同的model 每个模型的参数都保持一致 把数据拆分不同部分放进不同CPU
    #每个GPU计算出不同的loss和梯度
    #相当于每个GPU都扮演一个独立的进程 互不干扰
    #DDP使用分桶ringallreduce算法聚合所有gpu上的梯度 再把这个聚合后的梯度分发给每个GPU 对参数更新
    #分桶算法优势在于 它将梯度和通讯叠加 同步过程不同等待 GPU始终不闲置
    #这样的分发聚合 虽然每个GPU上的样本不同但整个过程中每个GPU的参数值完全相同
    #使用：分布式通信包 torch.distributed(核心是进程组的概念)
    #必须先注册端口号 表示先声明主机上的1个端口用于数据交换
    #常用通信后端backend NCCL(GPU环境) Gloo(CPU环境) MPI(通用后端标准)
    #torch.distributed.init_process_group(backend=,world_size=,rank=) 
    #其中 backend是通信后端标准 world_size是GPU个数 rank是把第0,...k块GPU中哪个作为主进程即主GPU
    #torch.multiprocessing 多进程专用包 创建多进程
    #torch.multiprocessing.spawn(函数名, args=(,,,), nprocs=,join=bool)
    #其中 函数名表示把这个函数作为一个单独的进程 args是这个函数要用的参数
    #nprocs是进程个数 join表示n个进程同时运行
    #函数里的rank参数要指定 使用时可以不赋值因为有系统子自动指定 表示0，...k个GPU进程
    #端口号最好4位以上 端口号范围0~65535
    #torch.utils.data.DistributedSampler(dataset,num_replicas,rank)分布式采样器 
    # 根据GPU进程对数据进行拆分 num_replicas是进程数即GPU个数
    #用Trainer 现在TrainingArguments(ddp_find_unused_parameters=False,
    # local_rank=os.environ['RANK'],fp16=True,lr_scheduler_type='',
    # warmup_steps=k) 
    # local_rank是当前进程RANK号 ddp_f_u_p是优化DDP性能 fp16是使用混合精度
    # lr_scheduler_type设置动态学习率 可以是='linear' 设置线性动态学习率
    # warmup_steps设置动态学习率的warmup阶段步数为k