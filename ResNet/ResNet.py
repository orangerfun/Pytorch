import torch
from torch import nn, optim
import torch.nn.functional as F
from BatchNormalizationZero import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
	def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
		super(Residual, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		# 如果上面的卷积改变了输出的通道数，则使用后use1x1conv改变x的通道数，从而可以相加
		if use_1x1conv:
			self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
		else:
			self.conv3 = None
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		y = F.relu(self.bn1(self.conv1(x)))
		y = self.bn2(self.conv2(y))
		if self.conv3:
			x = self.conv3(x)
		return F.relu(y+x)

class GlobalAvgPool2d(nn.Module):
	def __init__(self):
		super(GlobalAvgPool2d, self).__init__()
	def forward(self, x):
		return F.avg_pool2d(x, kernel_size=x.size()[2:])

# 起始主网络
net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
					nn.BatchNorm2d(64),
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
	if first_block:
		assert in_channels == out_channels
	blk = []
	for i in range(num_residuals):
		if i == 0 and not first_block:
			blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
		else:
			blk.append(Residual(out_channels, out_channels))
	return nn.Sequential(*blk)

# 加入残差块
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

# 加入全局平均池化和全连接层
net.add_module("global_avg_pool", GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

if __name__ == '__main__':
	batch_size=256
	train_iter, test_iter=load_data_fashion_mnist(batch_size, resize=96)
	lr, num_epochs = 0.001, 5
	optimizer = optim.Adam(net.parameters(), lr=lr)
	train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
