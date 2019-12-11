import numpy as np
import torch
from torch import nn
from torch.nn import init

num_train, num_test, n_input, batch_size, lr = 20, 100, 200, 5, 0.01
data = torch.tensor(np.random.normal(0,1,(num_train+num_test, n_input)), dtype=torch.float)
w_true, b_true = torch.arange(1, n_input+1, dtype=torch.float).view(-1,1), 3.0
label = torch.mm(data, w_true) + b_true
label += torch.randn(label.size())
# print(data.shape, label.shape)

train_data, train_label = data[:num_train], label[:num_train]
test_data, test_label = data[num_train:], label[num_train:]

train_d = torch.utils.data.TensorDataset(train_data, train_label)
test_d = torch.utils.data.TensorDataset(test_data, test_label)
train_iter = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=True)

net = nn.Sequential(nn.Linear(n_input, 1))
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=0.1)

init.normal_(net[0].weight, mean=0, std=0.1)
init.normal_(net[0].bias, mean=1, std=1)

epochs = 1000
eposilon = 0.05
for epoch in range(epochs):
	l_sum,n = 0, 0
	for x, y in train_iter:
		y_hat = net(x)
		# l = loss(y_hat, y)+eposilon*torch.pow(net[0].weight,2).sum()/(len(x))
		l = loss(y_hat, y)
		l.backward()
		optimizer.step()
		optimizer.zero_grad()
		l_sum += l
		n += 1
	print("epoch:%d, \tloss:%.3f"%(epoch+1, l_sum/n))
print("final loss:%.3f"%(l_sum/n))
for test_x, test_y in test_iter:
	test_loss = loss(net(test_x), test_y)
	print("test_loss:%.3f"%test_loss)
	break


