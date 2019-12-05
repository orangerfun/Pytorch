# Pytorch
Pytorch深度学习<br>
使用torch实现机器学习到深度学网络
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
