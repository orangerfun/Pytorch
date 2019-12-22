# 目录
* 1.torch.Tensor
* 2.torch.index_select
* 3.tensor.sum
* 4.torch.gather
* 5.torch.max
* 6.torch.cat
* 7.torch.nn.CrossEntropyLoss
* 8.tensor.data & tensor.detach
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
### 5.torch.max(input, dim, keepdim=False, out=None)
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
### 6.torch.cat((tensorA, tensorB), dim=0)
根据dim指定的维数拼接tensorA,B<br>
```
tensorA = torch.ones(2,3)
tensorB = torch.zeros(2,3)
tensorC = torch.cat((tensorA, tensorB),dim=0)
print(tensorA)
print(tensorB)
print(tensorC)
print(torch.cat((tensorA,tensorB),dim=1))
```
result:
```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
        
tensor([[0., 0., 0.],
        [0., 0., 0.]])
        
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [0., 0., 0.],
        [0., 0., 0.]])
        
tensor([[1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.]])
```
### 7. loss = torch.nn.CrossEntropyLoss(size_average=False)
几点注意事项：（1）size_average=False表示对一个batch的损失求和，不求均值；等于True表示对一个batch的样本求出的losss是均值，默认是True<br>
(2) 传入loss(y_hat, y);其中y_hat是没有经过softmax的；y也没有one_hot
```python3
import torch
import numpy as np
def softmax(x):
	e_x = x.exp()
	exsum = e_x.sum(dim = 1, keepdim=True)
	return e_x/exsum

def cross_entropy_loss(y_hat, y):
	return -torch.log(y_hat.gather(dim=1, index=y.view(-1,1)))

pred = torch.tensor([[1,2,3], [2,3,2]],dtype=torch.float)
label = torch.tensor([1,1])
s_pred = softmax(pred)

print("自己计算:",cross_entropy_loss(s_pred, label))
print("自己计算+mean:", cross_entropy_loss(s_pred, label).mean())
print("自己计算+sum:",cross_entropy_loss(s_pred,label).sum())

loss = torch.nn.CrossEntropyLoss()
print("内置函数计算：",loss(pred, label))

loss = torch.nn.CrossEntropyLoss(size_average=False)
print("内置函数+size_average=False:", loss(pred, label))
```
result:
```
自己计算: tensor([[1.4076],
                 [0.5514]])
自己计算+mean: tensor(0.9795)
自己计算+sum: tensor(1.9591)
内置函数计算： tensor(0.9795)
内置函数+size_average=False: tensor(1.9591)
```
### 8. tensor.data & tensor.detach
**(1)tensor.data**<br>
`x.data` 返回和 x 的相同数据 tensor,而且这个新的tensor和原来的tensor是共用数据的，一者改变，另一者也会跟着改变，而且新分离得到的tensor的`require s_grad = False`, 即不可求导的
```python3
import torch
a = torch.tensor([1,2,3.], requires_grad = True)
out = a.sigmoid()
c = out.data  # 需要走注意的是，通过.data “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
c.zero_()     # 改变c的值，原来的out也会改变
print(c.requires_grad)
print(c)
print(out.requires_grad)
print(out)
print("----------------------------------------------")
out.sum().backward() # 对原来的out求导，
print(a.grad)  # 不会报错，但是结果却并不正确
'''运行结果为：
False
tensor([0., 0., 0.])
True
tensor([0., 0., 0.], grad_fn=<SigmoidBackward>)
----------------------------------------------
tensor([0., 0., 0.])
'''
```
**(2)tensor.detach**<br>
`x.detach() `返回和 x 的相同数据 tensor,而且这个新的tensor和原来的tensor是共用数据的，一者改变，另一者也会跟着改变，而且新分离得到的tensor的`require s_grad = False`, 即不可求导的
```python3
import torch
a = torch.tensor([1,2,3.], requires_grad = True)
out = a.sigmoid()
c = out.detach()  # 需要走注意的是，通过.detach() “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
c.zero_()     # 改变c的值，原来的out也会改变
print(c.requires_grad)
print(c)
print(out.requires_grad)
print(out)
print("----------------------------------------------")
out.sum().backward() # 对原来的out求导，
print(a.grad)  # 此时会报错，错误结果参考下面,显示梯度计算所需要的张量已经被“原位操作inplace”所更改了。
'''运行结果为：
False
tensor([0., 0., 0.])
True
tensor([0., 0., 0.], grad_fn=<SigmoidBackward>)
----------------------------------------------
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
'''
```
**(3)两者比较**<br>
从上面的例子可以看出，使用`tensor.data`时，由于我更改分离之后的变量值c,导致原来的张量out的值也跟着改变了，但是这种改变对于autograd是没有察觉的，它依然按照求导规则来求导，导致得出完全错误的导数值却浑然不知。它的风险性就是如果我再任意一个地方更改了某一个张量，求导的时候也没有通知我已经在某处更改了，导致得出的导数值完全不正确，故而风险大<br>
使用`tensor.detach`时，由于我更改分离之后的变量值c,导致原来的张量out的值也跟着改变了，这个时候如果依然按照求导规则来求导，由于out已经更改了，所以不会再继续求导了，而是报错，这样就避免了得出完全牛头不对马嘴的求导结果。


#  参考
本内容主要参考：【[动手学深度学习](http://zh.d2l.ai/chapter_natural-language-processing/index.html)】<br>
程序参考：(https://github.com/ShusenTang/Dive-into-DL-PyTorch)

