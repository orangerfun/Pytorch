import torch
import time
from BatchNormalizationZero import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Sequential(nn.Conv2d(1, 6, 5),
					nn.BatchNorm2d(6),
					nn.Sigmoid(),
					nn.MaxPool2d(2, 2),
					nn.Conv2d(6, 16, 5),
					nn.BatchNorm2d(16),
					nn.Sigmoid(),
					nn.MaxPool2d(2,2),
					FlattenLayer(),
					nn.Linear(16*4*4, 120),
					nn.BatchNorm1d(120),
					nn.Sigmoid(),
					nn.Linear(120, 84),
					nn.BatchNorm1d(84),
					nn.Sigmoid(),
					nn.Linear(84, 10)
					)

if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size)
	lr, num_epochs = 0.001, 50
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)