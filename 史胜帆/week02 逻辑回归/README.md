## 逻辑回归 
### 1.plot绘制梯度下降过程可视化图形 观察不同学习率下的梯度下降过程
### 2.sklearn.datasets make_classification 获取数据并model_selection train_test_split 按7；3数据切分 并完成一个二分类训练任务
### 3.sklearn.datasets load_iris 用鸢尾花数据集 取前100个样本是2classes train_test_split 数据7；3切分
#### 3.1 对切分后的鸢尾花数据集训练逻辑回归模型 激活函数用sigmoid 损失函数用伯努利分布的似然函数取log取-负
#### 3.2 采用不同学习率和数据切分比例 观察训练结果差异
#### 3.3 保存训练后的模型参数theta bias和测试数据集X_test y_test
#### 3.4 在另一个文件里加载模型参数和测试数据集 进行模型推理预测 结果显示模型推理能力良好
