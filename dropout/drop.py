import torch
from torch import nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

# 定义dropout
def dropout(X, drop_prob):
	X = X.float()
	assert 0<=drop_prob<=1
	keep_prob = 1-drop_prob
	if keep_prob == 0:
		return torch.zeros_like(X)
	mask = (torch.randn(X.shape) < keep_prob).float()
	return mask * X/keep_prob

def test_acc(net, data_iter):
	acc , n = 0, 0
	for x, y in data_iter:
		acc = 0
		n = 0
		y_hat = net(x, is_training=False)
		acc += (y_hat.argmax(dim=1)==y).float().sum().item()
		n += x.shape[0]
	return acc/n



# 定义模型参数
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hidden1)), dtype=torch.float,requires_grad = True)
# b1 = torch.tensor(np.random.normal(0, 1, (1,num_hidden1)), dtype=torch.float, requires_grad = True)
b1 = torch.zeros((1, num_hidden1),requires_grad = True)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hidden1, num_hidden2)), dtype = torch.float, requires_grad = True)
# b2 = torch.tensor(np.random.normal(0, 1, (1,num_hidden2)), dtype = torch.float, requires_grad = True)
b2 = torch.zeros((1, num_hidden2), requires_grad = True)
w3 = torch.tensor(np.random.normal(0, 0.01, (num_hidden2, num_outputs)), dtype = torch.float, requires_grad = True)
# b3 = torch.tensor(np.random.normal(0, 1, (1,num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros((1, num_outputs), requires_grad = True)
params = [w1, b1, w2, b2, w3,b3]

#定义模型:使用is_training来确认是否是训练
drop_prob1 = 0.2
drop_prob2 = 0.5
def net(x, is_training):
	x = x.view((-1, num_inputs))
	h1 = (torch.mm(x, w1) + b1).relu()
	if is_training:
		h1 = dropout(h1, drop_prob1)
	h2 = (torch.mm(h1, w2) + b2).relu()
	if is_training:
		h2 = dropout(h2, drop_prob2)
	return torch.mm(h2, w3) + b3

# 加载数据
batchsize = 256
train_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=False, transform = transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=False)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params, lr=0.1)


epochs = 100
for epoch in range(epochs):
	n = 0
	l_sum = 0
	train_acc = 0
	for x, y in train_iter:
		y_hat = net(x, is_training=True)
		l = loss(y_hat, y)
		# print(l)
		l.backward()
		optimizer.step()
		optimizer.zero_grad()
		n += x.shape[0]
		l_sum += l*x.shape[0]
		train_acc += (y_hat.argmax(dim=1)==y).float().sum().item()
		# print(y_hat.argmax(dim=1),"\t", y)
	test_acu = test_acc(net, test_iter)
	print("epoch:%d, \tloss:%.3f, \tacc:%.3f, test_acc:%.3f"%(epoch+1, l_sum/n, train_acc/n, test_acu))









