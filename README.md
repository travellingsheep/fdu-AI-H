# AI Course Projects

本仓库包含两个人工智能课程项目：

- `pj1`：围绕神经网络基础、图像分类和残差网络展开，包含从零实现 BP、使用 PyTorch 搭建 CNN、实现 ResNet 等任务。
- `pj2`：围绕序列标注和命名实体识别展开，包含 HMM、CRF、Transformer+CRF 以及基于模板的 CRF bonus。

项目说明主要整理自各实验报告与实验文档。

## PJ1：神经网络与图像分类

### Task 1：BP 神经网络基础

Task 1 主要完成两个基础实验：

- 使用手写 BP 神经网络拟合 `sin(x)` 函数。
- 将 MNIST 手写数字图片展平成向量，使用全连接神经网络完成分类。

该任务重点在于理解并实现神经网络的完整训练流程，包括参数初始化、前向传播、反向传播、loss function 选择、学习率调整、mini batch 训练和可视化分析。`sin(x)` 拟合使用 MSE 作为损失函数，MNIST 分类使用 cross entropy，并通过 `interview()` 接口支持模型复核。

实验中比较了采样点数量、网络层数、神经元数量、学习率、训练轮数和 batch size 对模型效果的影响。最终 `sin(x)` 使用单隐藏层取得较好拟合效果，MNIST 使用两层全连接网络取得约 96% 以上的准确率。

### Task 2：CNN 图像分类

Task 2 使用 PyTorch 搭建不含 residual 结构的 CNN，分别完成 MNIST 和 CIFAR-10 分类。

MNIST 部分采用两层卷积层和两层线性层，流程为卷积、激活函数、最大池化、展平和全连接分类。实验对是否使用 `Normalize`、卷积层数和线性层数进行了比较，最终两层卷积加两层线性层的结构整体表现更稳，准确率可达到约 99%。

CIFAR-10 部分使用更深的浅层 CNN，并加入数据增强，包括随机裁剪、随机水平翻转、张量转换和归一化。实验比较了三层卷积、四层卷积、训练轮数和 dropout 的影响。结果显示，在浅层网络中 dropout 并未带来提升，去掉 dropout 后四层卷积结构在 CIFAR-10 上达到约 82% 的准确率。

### Task 3：ResNet 与消融实验

Task 3 在 MNIST 和 CIFAR-10 上实现并测试 ResNet，重点理解 residual connection 对深层网络训练的帮助。

该任务讨论了学习率、batch size、权重初始化、卷积、池化、stride、激活函数、dropout 和优化器选择等超参数与结构设计。实验中使用残差块解决深层网络难以学习恒等映射的问题，并比较了不同深度的 ResNet。

在 MNIST 上，ResNet 很快达到 99% 以上准确率；在 CIFAR-10 上，约 14 层 ResNet 将准确率从普通 CNN 的 82% 左右提升到 90% 左右，约 20 层 ResNet 进一步超过 91%。

### Bonus：防止过拟合

PJ1 的 bonus 总结了常见防止过拟合方法，包括数据增强、正则化、batch normalization、dropout、weight decay、早停、数据清洗和集成学习。实验结论强调，不同正则化方法需要结合模型深度和任务复杂度选择，浅层模型中盲目使用 dropout 可能反而降低效果。

## PJ2：命名实体识别

### Task 1：HMM 序列标注

Task 1 使用 HMM 隐式马尔可夫模型完成中英文 NER 任务。

模型通过训练集统计初始概率、转移概率和发射概率，并使用 Viterbi 解码得到最优标签序列。实验中加入 Laplace smoothing，避免未出现过的标签、转移或发射概率导致概率链断裂。

该任务重点比较了 smoothing 参数和 label 集合构造方式的影响。最终中文在 `smoothing=1.1` 时 F1 约为 0.8912，英文在 `smoothing=0.01` 时 F1 约为 0.7876。实验也指出 HMM 由于单向建模和 label bias，模型上限较低。

### Task 2：CRF 与平均感知机

Task 2 使用自定义 CRF 感知机模型完成 NER。

模型记录二元转移参数和标签-词发射参数，通过 Viterbi 解码进行预测。相比 HMM，CRF 使用分数累加而非概率连乘，并显式建模句子的开始与结束状态。英文任务中还额外加入首字母大小写特征。

训练初期直接在线硬更新会导致分数不稳定，实验最终采用平均参数方式缓解过拟合和后期参数覆盖问题。英文 F1 最高约 0.88，中文 F1 最高约 0.94，相比 HMM 有明显提升。

### Task 3：Transformer+CRF

Task 3 将 Transformer encoder 与 CRF 结合，用深度学习方法完成 NER。

代码中实现了数据集加载、词表和标签映射、位置编码、Transformer 编码器、CRF 负对数似然损失、partition function、真实路径得分计算和 Viterbi 解码。实验将最大句长设为 256，以覆盖验证集中的长句，并采用 `d_model=64`、`nhead=8`、`num_layers=4` 的结构以降低训练成本。

实验发现，较大的 `d_model=512` 在当前数据规模和设备条件下训练开销过高且效果不理想；改为 `d_model=64` 后，中文 F1 约为 0.8605，英文 F1 约为 0.7782。结果说明在小数据集 NER 任务中，手工特征更充分的 CRF 仍然可能比 Transformer+CRF 更高效。

### Bonus：模板 CRF

Bonus 基于给定的 `template_for_crf.utf8` 实现模板解析和特征抽取，将 unigram、bigram 等模板统一纳入 CRF 特征系统。

实验表明，加入更多模板并不必然提升分数。中文在使用 unigram 模板时很快达到约 0.94，英文在加入 unigram 和 bigram 后约为 0.83，但训练时间显著增加。结论强调模板应重视质量而不是数量，盲目扩大特征和计算量可能造成稀疏性上升、训练变慢和效果下降。

## Project Tree

```text
.
├── pj1
│   ├── week2lab.md
│   ├── pj1总结.txt
│   ├── pj1提交细则.md
│   ├── test_set
│   │   ├── t10k-images.idx3-ubyte
│   │   ├── t10k-labels.idx1-ubyte
│   │   └── test_batch
│   └── all_code
│       ├── task1
│       │   ├── 实验文档.pdf
│       │   ├── sin(x)拟合
│       │   │   ├── my_code.py
│       │   │   ├── mymodel.py
│       │   │   ├── test.py
│       │   │   └── model.pickle
│       │   └── mnist手写数字识别
│       │       ├── my_code.py
│       │       ├── mymodel.py
│       │       ├── test.py
│       │       └── model.pickle
│       ├── task2
│       │   ├── 实验报告.pdf
│       │   ├── bonus-防止过拟合的方法.pdf
│       │   ├── bonus-不使用pytorch完成CNN，但是只实现了部分内容.py
│       │   ├── mnist
│       │   │   ├── my_code.py
│       │   │   ├── mymodel.py
│       │   │   ├── test.py
│       │   │   └── model.pickle
│       │   └── cifar-10
│       │       ├── my_code.py
│       │       ├── mymodel.py
│       │       ├── test.py
│       │       └── model.pickle
│       └── task3
│           ├── 实验文档.pdf
│           ├── mnist
│           │   ├── my_code.py
│           │   ├── mymodel.py
│           │   ├── test.py
│           │   └── model.pickle
│           └── cifar-10
│               ├── my_code.py
│               ├── mymodel.py
│               ├── test.py
│               └── model.pickle
└── pj2
    ├── 实验文档.pdf
    ├── dataset_and_template
    │   ├── template_for_crf.utf8
    │   ├── Chinese
    │   │   ├── train.txt
    │   │   ├── validation.txt
    │   │   └── tag.txt
    │   └── English
    │       ├── train.txt
    │       ├── validation.txt
    │       └── tag.txt
    ├── task1
    │   ├── mycode.py
    │   ├── inference.py
    │   ├── Chinese_model.pt
    │   ├── English_model.pt
    │   ├── Chinese_output.txt
    │   └── English_output.txt
    ├── task2
    │   ├── mycode.py
    │   ├── inference.py
    │   ├── Chinese_model.pkl
    │   ├── English_model.pkl
    │   └── output.txt
    ├── task3
    │   ├── mycode.py
    │   ├── inference.py
    │   ├── Chinese_model.pth
    │   ├── English_model.pth
    │   └── new_validation_prediction.txt
    └── bonus
        ├── use_template.py
        ├── use_template_inference.py
        ├── Chinese_model.pkl
        └── English_model.pkl
```
