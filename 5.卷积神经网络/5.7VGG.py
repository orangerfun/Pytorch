import time
import torch
from torch import nn, optim
import torchvision
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def vgg_block(num_convs, in_channels, out_channels):
	blk = []
	for i in range(num_convs):
		if i == 0:
			blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
		else:
			blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
	blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
	return nn.Sequential(*blk)


conv_arch = ((1,1,64),(1,64,128),(2, 128, 256),(2, 256, 512),(2, 512, 512))
fc_features = 512*7*7
fc_hidden_units = 4096


class FlattenLayer(nn.Module):
	"""docstring for FlattenLayer"""
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
	net = nn.Sequential()
	for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
		net.add_module("vgg_block_"+str(i+1), vgg_block(num_convs, in_channels, out_channels))
	net.add_module("fc", nn.Sequential(FlattenLayer(), 
										nn.Linear(fc_features, fc_hidden_units),
										nn.ReLU(),
										nn.Dropout(0.5),
										nn.Linear(fc_hidden_units, fc_hidden_units),
										nn.ReLU(),
										nn.Dropout(0.5),
										nn.Linear(fc_hidden_units, 10)
										))
	return net

# net = vgg(conv_arch, fc_features, fc_hidden_units)
# X = torch.rand(1,1,224,224)
# for name, blk in net.named_children():
# 	X = blk(X)
# 	print(name, "output shape:", X.shape)


# 获取数据和训练模型
ratio = 8
small_conv_arch = [(1,1,64//ratio), (1, 64//ratio, 128//ratio),(2, 128//ratio, 256//ratio),
					(2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features//ratio, fc_hidden_units//ratio)
# print(net)

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

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, 224)
# for x, y in test_iter:
# 	print(x.shape, y.shape)
# 	break


def evaluate_accuracy(data_iter, net, device):
	acc_sum = 0
	n = 0
	for x, y in data_iter:
		if isinstance(net, nn.Module):
			net.eval()
			acc_sum += (net(x.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
			net.train()
		else:
			if "is_training" in net.__code__.co_varnames:
				acc_sum += (net(x.to(device), is_training=False).argmax(dim=1)==y.to(device)).float().sum().cpu().item()

			else:
				acc_sum += (net(x.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
		n += x.shape[0]
	return acc_sum/n

def train(net, train_iter, test_iter, batch_size, optimizer, device, epochs):
	net = net.to(device)
	loss = nn.CrossEntropyLoss()
	for epoch in range(epochs):
		train_acc_sum, train_l_sum, n, start = 0, 0, 0, time.time()
		batch_count = 0
		for x, y in train_iter:
			y_hat = net(x.to(device))
			l = loss(y_hat, y.to(device))			
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_l_sum += l
			train_acc_sum += (y_hat.argmax(dim=1)==y.to(device)).float().sum().cpu().item()
			n += x.shape[0]
			batch_count += 1
		test_acc = evaluate_accuracy(test_iter, net, device)
		print("epoch:%d\tloss:%.3f\ttrain_acc:%.3f\ttest_acc:%.3f\tspend_time:%.3f"%(epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))

lr = 0.01
epochs = 5
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, epochs)

