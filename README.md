# Pytorch
Pytorch深度学习<br>
使用torch实现机器学习到深度学网络,主要包括以下内容：<br>
* 1.线性回归
* 2.softmax回归
* 3.多层感感知机
# torch 基础
### 1. torch.Tensor
![](https://github.com/orangerfun/Pytorch/raw/master/tensor.png)
`torch.Tensor`是默认的tensor类型(`torch.FloatTensor`）的简称<br>
### 2. torch.index_select(x, dim , indices)
在数据x中按照dim指定的维度选出indice指定的数据，例如：
```python3
import torch
x = torch.linspace(1, 12, steps=12).view(3,4)
print(x)

indices = torch.LongTensor([0, 2])
y = torch.index_select(x, 0, indices)
print(y)
 
z = torch.index_select(x, 1, indices)
print(z)
 
z = torch.index_select(y, 1, indices)
print(z)
```
结果：
```
x:      tensor([[  1.,   2.,   3.,   4.],
                [  5.,   6.,   7.,   8.],
                [  9.,  10.,  11.,  12.]])
        
y:      tensor([[  1.,   2.,   3.,   4.],
                [  9.,  10.,  11.,  12.]])
                
z:      tensor([[  1.,   3.],
                [  5.,   7.],
                [  9.,  11.]])
                
z:     tensor([[  1.,   3.],
              [  9.,  11.]])
````
### 3.按维度求和
给定⼀个 Tensor 矩阵 X 。我们可以只对其中同⼀列（ dim=0 ）或同⼀⾏（ dim=1 ）的元素求和，并在结果中保留⾏和列这两个维度（ keepdim=True ）
```
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))
```
result:
```
tensor([[5, 7, 9]])
tensor([[ 6],
 [15]])
```
**大多数情况下，dim = 1表示在一行中求均值/求和等；而dim = 0则表示在一列中求...**

### 4.torch.gather(input, dim, index)
沿给定轴dim，将输入索引张量index指定位置的值进行聚合<br>
也可写成input.gather(dim, index)
```
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))    #dim =1 表示从每行中取数
```
result:
```
tensor([[0.1000],
 [0.5000]])

```
### torch.max(input, dim, keepdim=False, out=None)
按维度dim 返回最大值和对应的索引<br>
* torch.max()[0]， 只返回最大值的每个数
* troch.max()[1]， 只返回最大值的每个索引
* torch.max()[1].data 只返回variable中的数据部分（去掉Variable containing:）
* torch.max()[1].data.numpy() 把数据转化成numpy ndarry
* torch.max()[1].data.numpy().squeeze() 把数据条目中维度为1 的删除掉
* **torch.max(input=tensor1,other=tensor2) element-wise 比较tensor1 和tensor2 中的元素，返回较大的那个值**
```python3
import torch
import numpy as np
x = torch.tensor(np.random.normal(0,1,(2,3)), dtype=torch.float32)
print(x)
print(torch.max(input=x, dim=1))
print(torch.max(input=x, dim=1)[0].numpy())
print(torch.max(input=x,other=torch.tensor(0.0)))
```
result:
```
tensor([[-1.2088, -0.9236,  2.1777],
        [-1.3959,  0.3768,  0.3697]])

torch.return_types.max(
values=tensor([2.1777, 0.3768]),
indices=tensor([2, 1]))

[2.1776507 0.3768374]

tensor([[0.0000, 0.0000, 2.1777],
        [0.0000, 0.3768, 0.3697]])
```
#  reference
本内容主要参考：【[动手学深度学习](http://zh.d2l.ai/chapter_natural-language-processing/index.html)】<br>
程序参考：(https://github.com/ShusenTang/Dive-into-DL-PyTorch)

