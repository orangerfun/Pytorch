import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
from VGG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Inception(nn.Module):
	def __init__(self, in_c, c1, c2, c3, c4):
		super(Inception, self).__init__()
		# 线路1, 单1x1卷积层
		self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
		# 线路2, 1x1卷积后接3x3卷积层
		self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
		# 线路3, 1x1卷积后接5x5卷积层
		self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
		# 线路4,3x3最大化池化层后接1x1卷积层
		self.p4_1 = nn.MaxPool2d(kernel_size = 3, stride=1, padding=1)
		self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		return torch.cat((p1, p2, p3, p4), dim=1)

class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	def forward(self, x):
		return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
	def __init(self):
		super(GlobalAvgPool2d, self).__init__()
	def forward(self, x):
		return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 第一个BLOCK, 64channels, 7x7conv2d
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# second block: 64 channels, 1x1 conv2d + 192channels, 3x3conv2d
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
					nn.Conv2d(64, 192, kernel_size=3, padding=1),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# third block:
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
					Inception(256, 128, (128,192), (32, 96), 64),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# fourth block:
b4 = nn.Sequential(Inception(480, 192, (96,208), (16,48), 64),
					Inception(512, 160, (112,224), (24,64), 64),
					Inception(512, 128, (128,256), (24,64), 64),
					Inception(512, 112, (144, 288), (32, 64), 64),
					Inception(528, 256, (160, 320), (32, 128), 128),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# fifth block
b5 = nn.Sequential(Inception(832, 256, (160,320), (32,128), 128),
					Inception(832, 384, (192, 384), (48, 128), 128),
					GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.05, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)





