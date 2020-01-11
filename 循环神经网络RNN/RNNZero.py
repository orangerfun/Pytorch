import torch
import math
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import DataPreprocess as DP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corpus_indices, char_to_idx, idx_to_char, vocab_size = DP.load_data_jay_lyrics()


# 将词表示成one_hot向量形式
def one_hot(x, n_class, dtype=torch.float32):
	'''
	:param n_class: 相当于vocab_size
	'''
	x = x.long()
	res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
	res.scatter_(1, x.view(-1, 1), 1)
	return res

# x = torch.tensor([0, 2])
# print(one_hot(x, vocab_size))

# 将一个矩阵one-hot
def to_onehot(x, n_class):
	return [one_hot(x[:,i], n_class) for i in range(x.shape[1])]


# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print("will use", device)

def get_params():
	def _one(shape):
		ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
		return torch.nn.Parameter(ts, requires_grad=True)   #parameter是可以迭代更新的
	# 隐藏层参数
	W_xh = _one((num_inputs, num_hiddens))
	W_hh = _one((num_hiddens, num_hiddens))
	b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
	W_hq = _one((num_hiddens, num_outputs))
	b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
	return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


# 定义模型

# 返回初始化的隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
	return (torch.zeros((batch_size, num_hiddens), device=device),)


# 计算一个时间步中的隐藏状态和输出
def rnn(inputs, state, params):
	'''
	inputs和outputs都是num_steps个shape=[batchsize, vocabsize]的矩阵
	'''
	W_xh, W_hh, b_h, W_hq, b_q = params
	H, = state   # H后面的逗号表示是元组形式
	outputs = []
	for X in inputs:
		H = torch.tanh(torch.matmul(X, W_xh)+torch.matmul(H,W_hh)+b_h)
		Y = torch.matmul(H, W_hq) + b_q
		outputs.append(Y)
	return outputs, (H,)

# 测试
# x = torch.arange(10).view(2,5)
# state = init_rnn_state(x.shape[0], num_hiddens, device)
# inputs = to_onehot(x.to(device), vocab_size)
params = get_params()
# outputs, state_new = rnn(inputs, state, params)
# print(len(outputs), outputs[0].shape, state_new[0].shape)


# 定义预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
	'''
	args: prefix:有数个字符的字符串（作为输入）
		  num_chars:被预测字符的个数
	func: 输入prefix预测后面num_chars个字符
	'''
	state = init_rnn_state(1, num_hiddens, device)
	output = [char_to_idx[prefix[0]]]
	for t in range(num_chars+len(prefix)-1):
		x = to_onehot(torch.tensor([[output[-1]]], device=device),vocab_size)
		(y, state) = rnn(x, state, params)
		if t < len(prefix)-1:
			output.append(char_to_idx[prefix[t+1]])
		else:
			output.append(int(y[0].argmax(dim=1).item()))
	return " ".join([idx_to_char[i] for i in output])

# 测试
# res=predict_rnn("中国",10, rnn, params,init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx)
# print(res)


# 裁剪梯度
def grad_clipping(params, theta, device):
	norm = torch.tensor([0.0], device=device)
	for param in params:
		norm += (param.grad.data**2).sum()
	norm = norm.sqrt().item()
	if norm > theta:
		for param in params:
			param.grad.data *= (theta/norm)


def sgd(params, lr, batchsize):
	for param in params:
		param.data -= lr*param.grad/batchsize


# 定义模型训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices,\
						  idx_to_char, char_to_index, is_random_iter, num_epochs, num_steps, lr, clipping_theta,\
						  batch_size, pred_period, pred_len, prefixs):
	# 如果随机采样
	if is_random_iter:
		data_iter_fn = DP.data_iter_random
	else:
		data_iter_fn = DP.data_iter_consecutive
	params = get_params()
	loss = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		if not is_random_iter:
			state = init_rnn_state(batch_size, num_hiddens, device)
		l_sum, n, start = 0.0, 0, time.time()
		data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
		for x, y in data_iter:
			if is_random_iter:
				state = init_rnn_state(batch_size, num_hiddens, device)
			else:
				for s in state:
					s.detach_()
			inputs = to_onehot(x, vocab_size)    # x.shape = [batch_size, num_step]; inputs.shape = [num_step, batch_size, vocab_size]
			(outputs, state) = rnn(inputs, state, params)    # output.shape =  [num_step, batch_size, vocab_size] 注意，这里并不是三维矩阵，是一个列表，元素是batch_size*vocatsize的矩阵
			outputs = torch.cat(outputs, dim=0)   # output.shape = [num_step*batch_size, vocab_size]
			Y = torch.transpose(y, 0, 1).contiguous().view(-1)
			l = loss(outputs, Y.long())
			if params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()
			l.backward()
			grad_clipping(params, clipping_theta, device)
			sgd(params, lr, 1)
			l_sum += l.item()*y.shape[0]
			n += y.shape[0]
		if (epoch+1)%pred_period == 0:
			print("epoch:%d, perplexity:%f, time:%.2f sec "%(epoch+1, math.exp(l_sum/n), time.time()-start))
			for prefix in prefixs:
				print("-", predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


# 训练模型
if __name__ == "main":
	num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
	pred_period, pred_len, prefixes = 50, 50, ["分开", "不分开"]

	train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, True, num_epochs,\
					  num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
