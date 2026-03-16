# Project1提交细则

## 模型提交细则

三个task都需要提交一个模型文件的pickle文件，助教复核时会使用以下代码，所以请使模型支持**interview()**接口：

```python
import pickle

# 'cifar10_testdata/test_batch' for task2，task3
# ['mnist_testdata/t10k-images.idx3-ubyte', 'mnist_testdata/t10k-labels.idx1-ubyte'] for task1
eval_datafile_path = ''
# task1.pickle、task2.pickle、task3.pickle
model_pickle_path = ''

# 读取保存的模型
with open(model_pickle_path, 'rb') as f:
    model = pickle.load(f)

test_accuracy = model.interview(eval_datafile_path)
print(f"测试准确率: {test_accuracy:.2f}%")
```

## 准确率得分细则

准确率得分在数据集上的占比为

1. task1 MLP  MNIST（100%）
2. task2 CNN(无residual结构) MNIST（20%）CIFAR-10（80%）
3. task3 resnet MNIST（20%）CIFAR-10（80%）

模型无法收敛且代码不成体系得0分

模型无法收敛但能给出成体系的代码得准确率得分（20分）的50%，10分

能收敛的准确率根据数据集进行排名，对名次按数据集的占比加权求和再排名得到总名次，前30%满分，后70%根据排名均匀获得10分到20分。

后70%的分数计算公式为：
$$
n为后70\%的总人数\\
i为名次，i\in[1,n]\\
得分=\frac{i+1-2n}{1-n}*10
$$
