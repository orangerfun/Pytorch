import torch
import numpy as np
from torch.nn import init
from torch import nn
import torchvision
from collections import OrderedDict
import torchvision.transforms as transforms


def evaluate_acc(net, data_iter):
	acc, n = 0, 0
	for x,y in data_iter:
		y_hat = net(x)
		acc += (y_hat.argmax(dim = 1)==y).float().sum().item()
		n += y.shape[0]
	return acc/n


# 加载数据
batch_size = 256
mnist_train = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=False, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

num_inputs = 784
num_outputs = 10


# 定义模型(way1)
# class LinearNet(nn.Module):
# 	 def __init__(self,num_inputs, num_outputs):
# 	 	super(LinearNet,self).__init__()
# 	 	self.linear = nn.LinearNet(num_inputs, num_outputs)
# 	 def forward(self, x):
# 	 	y = self.linear(x.view(x.shape[0],-1))
# 	 	return y
# net = LinearNet(num_inputs, num_outputs)


# x形状转换
class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer,self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)

# 模型定义（way2）
# net = nn.Sequential(OrderedDict([("flatten",FlattenLayer()),\
# 								("linear",nn.Linear(num_inputs, num_outputs))]))

# 模型定义（way3)
net = nn.Sequential(FlattenLayer(),\
					 nn.Linear(num_inputs, num_outputs))


# 初始化模型参数(对way3)
init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0)
# 对way2
# init.normal_(net.linear.weight, mean=0, std=0.01)
# init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.CrossEntropyLoss()


# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
	train_loss_sum, train_acc_sum, n = 0, 0, 0
	for x,y in train_iter:
		y_hat = net(x)
		l = loss(y,y_hat).sum()
		# 梯度清零
		optimizer.zero_grad()
		l.backward()
		optimizer.step()

		train_loss_sum += l.item()
		train_acc_sum += (y_hat.argmax(dim = 1)==y).float().sum().item()
		n += y.shape[0]
	test_acc = evaluate_acc(net, test_iter)
	print("epoch: %d, \ttrain_loss:%.3f, \ttrain_acc:%.3f, \ttest_acc:%.3f"\
			%(epoch, train_loss_sum/n, train_acc_sum/n, test_acc))













