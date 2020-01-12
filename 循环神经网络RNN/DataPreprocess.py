import torch
import random

def load_data_jay_lyrics():
	with open("./Datasets/jaychou_lyrics.txt", "r", encoding="utf-8") as f:
		corpus_chars = f.read()
	corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
	# 使用前10000个字符训练模型
	corpus_chars = corpus_chars[:10000]

	# 建立字符索引
	idx_to_char = list(set(corpus_chars))
	char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
	vocab_size= len(idx_to_char)

	# 将训练集中每个字符转化成索引
	corpus_indices = [char_to_idx[char] for char in corpus_chars]
	# sample = corpus_indices[:20]
	# print("chars:", " ".join([idx_to_char[idx] for idx in sample]))
	# print("indices:", sample)
	return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
	'''
	args: corpus_indices: list, 转换成索引的样本,整个文本组成的string
		  batch_size: 一个batch包含样本数
		  num_steps: 一个样本序列长度
	func: 将string切成长度为num_step的小段，随机选取batch_size个小段组成一个batch
	'''

	# 计算有多少样本， 减1是因为输出比输入延迟一个时刻
	num_examples = (len(corpus_indices)-1)//num_steps
	epoch_size = num_examples//batch_size    # 相当于num_batch
	example_indices = list(range(num_examples))
	random.shuffle(example_indices)

	def  _data(pos):
		'''
		func: 返回从pos开始的长为num_steps的序列
		'''
		return corpus_indices[pos:pos+num_steps]

	if device == None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	for i in range(epoch_size):
		i = i*batch_size
		batch_indices = example_indices[i:i+batch_size]
		x = [_data(j*num_steps) for j in batch_indices]
		y = [_data(j*num_steps+1) for j in batch_indices]
		yield torch.tensor(x, dtype=torch.float32, device=device),torch.tensor(y, dtype=torch.float32, device=device)

# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
	'''
	func: 先将整个文本组成的String平均分成batch_size分，并组成矩阵[batch_size, len],
		将矩阵在行维度上，按长度为num_step分割成小矩阵，每个小矩阵就是一个batch
		这样前一个batch和后一个batch在行维度上是有连续信息的
	'''
	if device == None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
	data_len = len(corpus_indices)
	batch_len = data_len//batch_size  # 分成batch_size份
	indices = corpus_indices[0:batch_size*batch_len].view(batch_size, batch_len)
	epoch_size = (batch_len-1)//num_steps
	for i in range(epoch_size):
		i *= num_steps
		x = indices[:, i:i+num_steps]
		y = indices[:, i+1:i+num_steps+1]
		yield x, y

# 测试
# myseq = list(range(30))
# for x, y in data_iter_consecutive(myseq, batch_size=2, num_steps=6):
# 	print("x:",x,"\n","y:", y,"\n")





