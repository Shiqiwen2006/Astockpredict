# Astockpredict

### 这是作业-A股预测
仓库内的文件：
- main(6).ipynb 题干和运行代码（在最后几个cell里面）
- main.py 运行代码（从main.ipynb里面拿出来的）
- test(1).py 训练脚本
- train_data.npy 训练数据
- results/mymodel.pt 我自己本地跑出来的模型

### 评价标准：

两个指标：MAE和MAPE

评分公式如下：

$$
\text{score} = (1 - 4 \cdot \mathrm{MAPE}) \times 40 + \left(1 - \frac{\sqrt{\mathrm{MAE}+1}}{3}\right) \times 60
$$

我用Claude写过一遍，现在的平台测试MAPE是0.0276，MAE是0.7675
