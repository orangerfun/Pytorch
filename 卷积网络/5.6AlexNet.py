import time
import torch
from torch import nn, optim
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(1, 96, 11,4),  #in_channels, out_channels, kernel_size, stride, padding
								nn.ReLU(),
								nn.MaxPool2d(3,2), #kernel_size, stride
								nn.Conv2d(96, 256, 5, 1, 2),
								nn.ReLU(),
								nn.MaxPool2d(3,2),
								nn.Conv2d(256, 384, 3, 1, 1),
								nn.ReLU(),
								nn.Conv2d(384, 384, 3, 1, 1),
								nn.ReLU(),
								nn.Conv2d(384, 256, 3, 1, 1),
								nn.ReLU(),
								nn.MaxPool2d(3,2)
								)
		self.fc = nn.Sequential(nn.Linear(256*5*5, 4096),
								nn.ReLU(),
								nn.Dropout(0.5),
								nn.Linear(4096, 4096),
								nn.ReLU(),
								nn.Dropout(0.5),
								nn.Linear(4096, 10))
	def forward(self, img):
		feature = self.conv(img)
		output = self.fc(feature.view(img.shape[0], -1))
		return output
net = AlexNet()


def load_data_fashion_mnist(batch_size, resize=None, root="./Datasets/FashionMNIST"):
	trans = []
	if resize:
		trans.append(torchvision.transforms.Resize(size=resize))
	trans.append(torchvision.transforms.ToTensor())

	transform = torchvision.transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
	mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)

	train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
	test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)
	return train_iter, test_iter

train_iter, test_iter = load_data_fashion_mnist(128, 224)
# for x, y in train_iter:
# 	print(x.shape, y.shape)
# 	exit()


def evaluate_accuracy(data_iter, net, device):
	acc_sum = 0
	n = 0
	for x, y in data_iter:
		if isinstance(net, nn.Module):
			net.eval()
			acc_sum += ((net(x.to(device)).argmax(dim=1))==y.to(device)).float().sum().cpu().item()
			net.train()
		else:
			if "is_training" in net.__code__.co_varnames:
				acc_sum += (net(x.to(device), is_training=False).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
			else:
				acc_sum += (net(x.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
		n += y.shape[0]
	return acc_sum/n

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
	net = net.to(device)
	loss = nn.CrossEntropyLoss()
	print("traing...wait..")
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n, start = 0, 0, 0, time.time()
		batch_count = 0
		for x, y in train_iter:
			x = x.to(device)
			y = y.to(device)
			y_hat = net(x)
			l = loss(net(x), y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_l_sum += l.cpu().item()
			train_acc_sum += (y_hat.argmax(dim=1)==y).float().sum().cpu().item()
			n += y.shape[0]
			batch_count += 1
		test_acc = evaluate_accuracy(test_iter, net, device)
		print("epoch:%d\tloss:%.3f\ttrain_acc:%.3f\ttest_acc:%.3f\ttime:%.3f"%(epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, start-time.time()))

lr, num_epochs, batch_size = 0.001, 5, 128
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)



