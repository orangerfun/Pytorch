import torch
from torch import nn, optim
import time
import torchvision
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps, momentum):
	# 判断预测还是训练， 预测时使用移动平均
	if not is_training:
		x_hat = (x-moving_mean)/torch.sqrt(moving_var+eps)
	else:
		assert len(x.shape) in (2, 4)
		# 全连接情况下
		if len(x.shape) == 2:
			mean = x.mean(dim=0)
			var = ((x-mean)**2).mean(dim=0)
		# 卷积情况下，对通道维度求均值
		else:
			mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
			var = ((x-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

		x_hat = (x-mean)/torch.sqrt(var+eps)
		moving_mean = momentum*moving_mean+(1-momentum)*mean
		moving_var = momentum*moving_var+(1-momentum)*var
	y_hat = gamma*x_hat + beta
	return y_hat, moving_mean, moving_var


def load_data_fashion_mnist(batch_size, resize=None, root="./Datasets/FashionMNIST"):
	trans = []
	if resize:
		trans.append(torchvision.transforms.Resize(size=resize))
	trans.append(torchvision.transforms.ToTensor())

	transform = torchvision.transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
	mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)

	train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
	test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
	return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device):
	acc, n = 0, 0
	for x, y in data_iter:
		x = x.to(device)
		y = y.to(device)
		if isinstance(net, nn.Module):
			net.eval()
			acc += (net(x).argmax(dim=1)==y).float().sum().cpu().item()
			net.train()
		else:
			if "is_training" in net.__code__.co_varnames:
				acc += (net(x, is_training=False).argmax(dim=1)==y).float().sum().cpu().item()
			else:
				acc += (net(x).argmax(dim=1)==y).float().sum().cpu().item()
		n += x.shape[0]
	return acc/n


def saveModel(net, path):
	torch.save(net.state_dict(), path)


def loadModel(net, path):
	new_net = net
	new_net.load_state_dict(torch.load(path))
	return new_net



def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
	# net = loadModel(net, "./savedModel/BatchNormalizationZero_model_16.ckpt")      #加载保存好的模型
	net = net.to(device)
	loss = torch.nn.CrossEntropyLoss()
	for epoch in range(num_epochs):
		train_l_sum, train_acc, n, start_time = 0, 0, 0, time.time()
		batchcount = 0
		for x, y in train_iter:
			x = x.to(device)
			y = y.to(device)
			y_hat = net(x)
			l = loss(y_hat, y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_l_sum += l
			train_acc += (y_hat.argmax(dim=1)==y).float().sum().cpu().item()
			n += x.shape[0]
			batchcount += 1
		test_acc = evaluate_accuracy(test_iter, net, device)
		print("epoch:%d\tmean_loss:%.3f\ttrain_acc:%.3f\ttest_acc:%.3f\ttime:%.3f\t"%(epoch+1, train_l_sum/batchcount, train_acc/n, test_acc, time.time()-start_time ))

		# 保存模型
		if epoch % 5 == 0:
			if not os.path.exists("./savedModel/"):
				os.mkdir("./savedModel/")
			saveModel(net, path="./savedModel/"+ os.path.basename(__file__).split(".")[0]+"_model_"+str(epoch+1)+".ckpt")
			print("model saved; epochs: %d"%(epoch+1))


# 定义一个batchnorm层
class BatchNorm(nn.Module):
	def __init__(self, num_features, num_dims):
		super(BatchNorm, self).__init__()
		# 如果是全连接，其形状如下，此处shape与求得的mean的shape相同
		if num_dims == 2:
			shape = (1, num_features)
		else:
			shape = (1, num_features, 1, 1)      #num_features相当于通道数
		self.gamma = nn.Parameter(torch.ones(shape))
		self.beta = nn.Parameter(torch.zeros(shape))
		self.moving_mean = torch.zeros(shape)
		self.moving_var = torch.zeros(shape)

	def forward(self, x):
		if self.moving_mean.device != x.device:
			self.moving_mean = self.moving_mean.to(x.device)
			self.moving_var = self.moving_var.to(x.device)
		# Module实例的training属性默认为True, 调用.eval()后设成false
		y, self.moving_mean, self.moving_var = batch_norm(self.training, x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
		return y


class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)



# 使用BN来训练LeNet网络
net = nn.Sequential(nn.Conv2d(1, 6, 5),
					BatchNorm(6, 4),
					nn.Sigmoid(),
					nn.MaxPool2d(2,2),
					nn.Conv2d(6, 16, 5),
					BatchNorm(16, num_dims=4),
					nn.Sigmoid(),
					nn.MaxPool2d(2,2), 
					FlattenLayer(),
					nn.Linear(16*4*4, 120),
					BatchNorm(120, 2), 
					nn.Sigmoid(),
					nn.Linear(120, 84),
					BatchNorm(84, 2), 
					nn.Sigmoid(),
					nn.Linear(84, 10))


if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size)
	lr, num_epochs = 0.001, 100
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

