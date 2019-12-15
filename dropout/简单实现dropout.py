import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn

batchsize = 256
num_inputs = 28*28
num_outputs = 10
n_hidden1 = 256
n_hidden2 = 256
drop_prob1 = 0.2
drop_prob2 = 0.5
epochs = 10
lr = 0.5

train_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=False, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=True)

def test_evaluate(data_iter, net):
	test_acc , n = 0, 0
	for x, y in data_iter:
		y_hat = net(x)
		test_acc += (y_hat.argmax(dim=1)==y).float().mean().item()
		n += 1
	return test_acc/n


class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)


net = nn.Sequential(FlattenLayer(),\
					nn.Linear(num_inputs, n_hidden1), \
					nn.ReLU(), \
					nn.Dropout(drop_prob1), \
					nn.Linear(n_hidden1, n_hidden2), \
					nn.ReLU(), \
					nn.Dropout(drop_prob2), \
					nn.Linear(n_hidden2, num_outputs))


# torch.nn.init.normal_(net[1].weight, mean=0, std=0.01)
# torch.nn.init.constant(net[1].bias, val=0)
# torch.nn.init.normal_(net[4].weight, mean=0, std=0.01)
# torch.nn.init.constant(net[4].bias, val=0)
for param in net.parameters():
	nn.init.normal_(param, mean=0, std=0.01)


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


for epoch in range(epochs):
	l_sum, n, train_acc = 0, 0, 0
	for x, y in train_iter:
		y_hat = net(x)
		l = loss(y_hat, y)
		l.backward()
		optimizer.step()
		optimizer.zero_grad()
		train_acc += (y_hat.argmax(dim=1)==y).sum().item()
		l_sum += l*x.shape[0]
		n += x.shape[0]
	test_acc = test_evaluate(test_iter, net)
	print("epoch:%d, \tloss:%.3f, \ttrain_acc:%.3f, \ttest_acc:%.3f"%(epoch, l_sum/n, train_acc/n, test_acc))



