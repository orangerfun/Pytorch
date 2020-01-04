## 1.程序说明
`BatchNormalizationZero.py`从零实现BatchNormalization, 并将其应用于AlexNet上<br>
`BatchNormalization.py`调用torch中接口`BatchNorm2d`和`BatchNorm1d`实现BN，并应用于AlexNet上
## 2.torch中BN接口
Pytorch 中 nn 模 块 定 义 的 `BatchNorm1d` 和 `BatchNorm2d` 类 使 ⽤ 起 来 更 加 简 单 ， ⼆ 者 分 别 ⽤ 于 全 连 接 层 和 卷 积 层 ， 都 需 要 指 定 输 ⼊ 的 `num_features` 参数值
```python3
nn.BatchNorm1d(num_features)     # num_features表示全链接层输出矩阵的列数
nn.BatchNorm2d(num_features)     # num_features代表卷积层的输出的通道数
```
## 3.小结
* 在模型训练时，批量归⼀化利⽤⼩批量上的均值和标准差，不断调整神经⽹络的中间输出，从⽽使整个神经⽹络在各层的中间输出的数值更稳定
* 批量归⼀化层和丢弃层⼀样，在训练模式和预测模式的计算结果是不⼀样的
