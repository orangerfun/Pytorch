# Intro
`mnist.py` 加载FashionMNIST数据集，程序包括：<br>
* `get_fashion_mnist_labels`将数值标签换成文本标签
* `show_fashion_mnist`画图
* 如何加载数据程序
<br>

`从零实现softmax.py`包括损失函数，随机梯度下降等都是手写，包括以下方面
* `softmax`定义softmax函数
* `net`定义网络
* `cross_entropy`定义交叉熵损失函数
* `evaluate_accuracy`定义模型准确率
* `sgd`定义随机梯度下降
* `train`定义训练函数
<br>

`softmax.py`损失函数、随机梯度下降等都是直接调库使用，包括以下方面：
* `FlatternLayer`x形状转换层
* `net`使用三种方式定义模型
* 初始画模型参数方法，损失函数定义方法
