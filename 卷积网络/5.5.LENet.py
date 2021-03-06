import time
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(1, 6, 5),\
								nn.Sigmoid(),\
								nn.MaxPool2d(2, 2),\
								nn.Conv2d(6, 16, 5),\
								nn.Sigmoid(),\
								nn.MaxPool2d(2, 2))
		self.fc = nn.Sequential(
						nn.Linear(16*4*4, 120),\
						nn.Sigmoid(),\
						nn.Linear(120, 84),\
						nn.Sigmoid(),\
						nn.Linear(84,10))
	def forward(self, x):
		feature = self.conv(x)
		output = self.fc(feature.view(x.shape[0],-1))
		return output
net = LeNet()
# print(net)

# 加载数据
batch_size = 256
import torchvision
import torchvision.transforms as transforms
train_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=False, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


def evaluate_accuracy(data_iter, net, device):
	acc_sum, n=0, 0
	with torch.no_grad():
		for x, y in data_iter:
			if isinstance(net, torch.nn.Module):
				net.eval()  #评估模式， 会关闭dropout
				acc_sum += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
				net.train() # 改回训练模式, 开启dropout
			else:  #如果是自定义模型
				if "is_training" in net.__code__.co_varnames:   # 如有is_traing这个参数
					acc_sum += (net(x.to(device), is_training=False).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
				else:
					acc_sum += (net(x.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
			n += y.shape[0]
	return acc_sum/n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
	net = net.to(device)
	print("training on:", device)
	loss = torch.nn.CrossEntropyLoss()
	batch_count = 0
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n, start = 0, 0, 0, time.time()
		for x, y in train_iter:
			x = x.to(device)
			y = y.to(device)
			y_hat = net(x)
			l = loss(y_hat, y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_l_sum += l.cpu().item()
			train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
			# print(train_acc_sum)
			n += y.shape[0]
			batch_count += 1
		test_acc = evaluate_accuracy(test_iter, net, device)
		print("epoch: %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f, time: %.4fsec"%(epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))


lr = 0.01
num_epochs = 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

