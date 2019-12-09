import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

batchsize = 256
train_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST", train=False, transform=transforms.ToTensor())
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=False)

# 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
w1 = torch.tensor(np.random.normal(0,0.01,(num_inputs, num_hiddens)),dtype=torch.float32)
b1 = torch.zeros(num_hiddens, dtype=torch.float32)
w2 = torch.tensor(np.random.normal(0, 0.01,(num_hiddens, num_outputs)), dtype=torch.float32)
b2 = torch.zeros(num_outputs,dtype=torch.float32)

params = [w1, b1, w2, b2]
for param in params:
	param.requires_grad_(requires_grad=True)

# 定义激活函数
def relu(x):
	return torch.max(input=x, other=torch.tensor(0.0))

# 定义模型
def net(x):
	x = x.view((-1, num_inputs))
	h = relu(torch.matmul(x,w1)+b1)
	return torch.matmul(h,w2)+b2

def test_evaluation(net, data_iter):
	acc, n =0, 0
	for x, y in data_iter:
		y_hat = net(x)
		acc += (y_hat.argmax(dim=1)==y).float().sum().item()
		n += y.shape[0]
	return acc/n

def SGD(params, lr, batchsize):
	for param in params:
		param.data -= lr*param.grad/batchsize


# 定义损失函数
loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.5)


num_epochs=5
for epoch in range(num_epochs):
	train_loss_sum, train_acc_sum, n = 0.,0.,0
	for x,y in train_iter:
		y_hat = net(x)
		l = loss(y_hat,y)     #注意crossEntropyLoss函数第一参数是y_hat, 第二个是y

		if params[0].grad is not None:    #要进行判断
			for param in params:
				param.grad.data.zero_()

		l.backward()
		SGD(params, 1, batchsize)

		train_loss_sum += l.item()*len(x)
		train_acc_sum += (y_hat.argmax(dim=1)==y).float().sum().item()
		n += len(x)

	test_acc = test_evaluation(net, test_iter)
	print("epoch:%d, \ttrain_loss:%.3f, \ttrain_acc:%.3f, \ttest_acc:%.3f"
		%(epoch+1, train_loss_sum/n, train_acc_sum/n, test_acc))

