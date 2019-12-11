import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# 使用较少的样本来表现出过拟合
n_train, n_test, n_inputs = 20, 100, 200
data = torch.tensor(np.random.normal(0, 1, (n_train+n_test, n_inputs)), dtype=torch.float)
w_true, b = torch.arange(1,201,dtype=torch.float).view(200,1), 4.5
label = torch.matmul(data, w_true) + torch.tensor(b)
label += torch.tensor(np.random.normal(0, 0.01, label.size()))

train_data, train_label = data[:n_train], label[:n_train]
test_data, test_label = data[n_train:], label[n_train:]

def data_iter(data, label, batchsize):
	num_examples = len(data)
	index = torch.arange(num_examples)
	random.shuffle(index)
	for i in range(0, num_examples, batchsize):
		indices = index[i:min(i+batchsize, num_examples)]
		yield data.index_select(0, indices), label.index_select(0, indices)

w = torch.tensor(np.random.normal(0, 0.1, (200,1)), dtype=torch.float)
b = torch.tensor(np.random.normal(0,1,1), dtype = torch.float)

def net(x, w, b):
	return torch.matmul(x, w) + b

def SGD(params, lr, batchsize):
	for param in params:
		param.data -= lr*param.grad/batchsize

def MLELOSS(y_pre, y_true):
	return torch.pow((y_pre-y_true),2)

def L2Norm(w):
	return torch.pow(w,2).sum()/2

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
epochs = 2000
batch_size = 5
lr = 0.001
epsilon = 1
for epoch in range(epochs):
	loss_sum, n = 0, 0
	for x,y in data_iter(train_data, train_label, batch_size):
		y_hat = net(x, w, b)
		loss = (MLELOSS(y_hat, y) + L2Norm(w)*epsilon).sum()
		loss.backward()
		SGD([w,b], lr, batch_size)
		w.grad.data.zero_()
		b.grad.data.zero_()
		loss_sum += loss.item()
		n += len(x)
	print("epoch:%d, \tloss:%.3f"%(epoch+1, loss_sum/n))
print("final loss: %.3f"%(loss_sum/n))

for x_test, y_test in data_iter(test_data,test_label,batch_size):
	loss_test = (MLELOSS(net(x_test, w, b), y_test)/batch_size).sum().item()
	print("test loss:%.3f"%loss_test)
	break


