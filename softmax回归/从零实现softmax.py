import torch
import torchvision
import numpy as np
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


batch_size = 256

# 加载数据
mnist_train = torchvision.datasets.FashionMNIST(root = "./Datasets/FashionMNIST",\
												train = True,\
												transform = transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root = "./Datasets/FashionMNIST",\
												train = False,\
												transform = transforms.ToTensor())
# 获取批次数据
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False)

# 初始化模型参数
num_inputs = 784
num_outputs = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float)
b = torch.zeros(num_outputs, dtype = torch.float)

# 设置允许模型参数梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义softmax函数
def softmax(x):
	x_exp = x.exp()
	sum_ = x_exp.sum(dim=1, keepdim = True)
	result = x_exp/sum_
	return result


# 定义网络
def net(x):
	return softmax(torch.mm(x.view(-1, num_inputs), w) + b)


# 定义交叉熵损失函数
def cross_entropy(y, y_hat):
	return -torch.log(y_hat.gather(dim=1, index=y.view(-1,1)))


# 计算准确率
"""
==返回的是True or False
.float将其变成float类型，0=False  1=True
.mean求均值
"""
def accuracy(y, y_hat):
	return (y_hat.argmax(dim=1) == y).float().mean().item()


# 定义模型的准确率
def evaluate_accuracy(net, data_iter):
	n, acc= 0, 0
	for x,y in data_iter:
		acc += (net(x).argmax(dim=1) == y).float().sum().item()
		n += y.shape[0]
	return acc/n


# 自定义的随机梯度下降
def sgd(params, lr, batch_size):
	for param in params:
		param.data -= lr*param.grad/batch_size

# 显示图像
def show_fashion_mnist(images, labels):
	_,figs = plt.subplots(1,len(labels), figsize=(12,12))
	for fig, lbs, imgs in zip(figs, labels, images):
		fig.imshow(imgs.view(28,28).numpy())
		fig.set_title(lbs)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
	plt.show()

# 标签文本化
def get_fashion_mnist_labels(labels):
	text_labels=["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker","bag", "ankleboot"]
	return [text_labels[int(i)] for i in labels]


# 训练模型
def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
	for epoch in range(num_epochs):
		train_loss_sum, train_acc_sum, n = 0, 0, 0
		for x,y in train_iter:
			y_hat = net(x)
			l = loss(y, y_hat).sum()

			# 梯度清零
			if optimizer is not None:
				optimizer.zero_grad()
			elif params is not None and params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()

			l.backward()
			if optimizer is None:
				sgd(params, lr, batch_size)
			else:
				optimizer.step()

			train_loss_sum += l.item()
			train_acc_sum += (y_hat.argmax(dim=1)==y).float().sum().item()
			n += y.shape[0]

		test_acc = evaluate_accuracy(net, test_iter)
		print("epoch:%d, \tloss:%.3f, \ttrain_acc:%.3f, \ttest_acc:%.3f"\
			%(epoch+1, train_loss_sum/n, train_acc_sum/n, test_acc))

if __name__ == "__main__":
	num_epochs , lr = 5, 0.1
	train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w,b], lr)

	# 预测：
	x, y = iter(test_iter).next()
	true_labels = get_fashion_mnist_labels(y.numpy())
	pred_labels = get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
	titles = [true+"\n"+pred for true,pred in zip(true_labels,pred_labels)]
	show_fashion_mnist(x[:9],titles[:9])

