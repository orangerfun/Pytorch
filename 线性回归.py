# -*- encoding:utf8 -*-
# @author orangerfun@gmail.com

import torch
import numpy as np
import random

# 生成数据
nums = 1000
length = 2
w_true = torch.tensor([2.5, -1.3],dtype = torch.float32).view(2,1)
b_true = 4.9

epsilon = torch.from_numpy(np.random.normal(0, 0.03, (nums,1)))
x = torch.tensor(np.random.normal(2, 0.3, (nums, length)), dtype=torch.float32)
y = torch.mm(x, w_true)+b_true+epsilon

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 1, (length, 1)), dtype = torch.float32)
b = torch.zeros(1, dtype = torch.float32)
# 后面要求梯度，因此设置requires_grad=True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
def linreg(x,w,b):
	return torch.mm(x,w)+b

# 自定义损失函数
def square_loss(y_hat, y):
	return (y_hat-y.view(y_hat.size()))**2/2

# 定义优化算法
def sgd(params, lr, batchsize):
	for param in params:
		param.data -= lr*param.grad/batchsize

# 读取数据
def data_iter(batchsize, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))
	random.shuffle(indices)
	for i in range(0, num_examples, batchsize):
		index = torch.LongTensor(indices[i:min(i+batchsize, num_examples)])
		yield features.index_select(0, index), labels.index_select(0, index)


# 训练模型
lr = 0.05
epochs = 100
batchsize = 10
net = linreg
loss = square_loss
for epoch in range(epochs):
	for x_train, y_train in data_iter(batchsize, x, y):
		l = loss(net(x_train, w, b),y_train).sum()
		l.backward()
		sgd([w,b], lr, batchsize)

		# 梯度清零
		w.grad.data.zero_()
		b.grad.data.zero_()

    #计算一个epoch下的总的loss
	train_l = loss(net(x, w, b), y)
	print("epoch=%d, \tmean_los=%.3f"%(epoch+1, train_l.mean().item()))

print("w1=%.3f, \tw2=%.3f"%(w[0], w[1]))
print("b=%.3f"%b)

print("w1_true=%.3f, \tw2_true=%.3f"%(w_true[0], w_true[1]))
print("b=%.3f"%b_true)


