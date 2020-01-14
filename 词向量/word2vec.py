import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data

assert "ptb.train.txt" in os.listdir("./Datasets/ptb")

# 读取数据
with open("./Datasets/ptb/ptb.train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]
    # print(len(raw_dataset))
    # for st in raw_dataset[-2:]:
    #     print(len(st), st)


# 建立词语索引；为了减少计算量保留数据中频率大于5的词
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
idx_to_token = [tk for tk, idx in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
# print(num_tokens)


# 二次采样
def discard(idx):
    t = 1e-4
    return random.uniform(0, 1) < 1-math.sqrt(t/(counter[idx_to_token[idx]]/num_tokens))

subsampled_dataset = [[tk for tk in st if not discard(tk)]for st in dataset]
# print(sum([len(st) for st in subsampled_dataset]))


# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i-window_size), min(len(st), i+window_size+1)))
            indices.remove(i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


# 测试中心词提取函数
# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print("data_set:", tiny_dataset)
# for centers, contexts in zip(*get_centers_and_contexts(tiny_dataset, 5)):
#     print("centers:", centers, "contexts:", contexts)

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# 负采样
def get_negtives(all_contexts, sampling_weights, K):
    all_negtives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for context in all_contexts:
        negtives = []
        while len(negtives) < len(context) * K:
            if i == len(neg_candidates):
                i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i+1
            if neg not in set(context):
                negtives.append(neg)
        all_negtives.append(negtives)
    return all_negtives


sampling_weights = [(counter[w]/num_tokens)**0.75 for w in idx_to_token]
all_negatives = get_negtives(all_contexts, sampling_weights, 5)
# print(all_contexts[:3])
# print(all_negatives[:3])


# 读取数据
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)


# 小批量数据格式化函数
def batchify(data):
    max_len = max(len(c)+len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0]*(max_len-cur_len)]
        masks += [[1]*cur_len + [0]*(max_len-cur_len)]
        labels += [[1]*len(context) + [0]*(max_len-len(context))]
    return torch.tensor(centers).view(-1,1), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels)


batch_size = 512
num_workers = 0 if sys.platform.startswith("win32") else 4

dataset = MyDataset(all_centers, all_contexts, all_negatives)

# 不能直接使用下方方法，因为all_contexts里面形状不规则
# dataset2 = Data.TensorDataset(torch.tensor(all_centers), torch.tensor(all_contexts), torch.tensor(all_negatives))

data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers)

# 测试
# for batch in data_iter:
#     for name, data in zip(["centers", "contexts_negatives", "masks", "labels"], batch):
#         print(name, "shape:", data.shape)
#     break

# 嵌入层
# embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
# print(embed.weight)
# x = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.long)
# print(embed(x))


# 跳字模型
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# 二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        '''
        :param inputs: shape=[batch_size, length]
        :param targets: the tensor of the same as inputs
        '''
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)
loss = SigmoidBinaryCrossEntropyLoss()


# pred = torch.tensor([[-1.0, 2],[0.5, 1]])
# label = torch.tensor([[1, 0.0],[2,5]])
# mask = torch.tensor([[1,0], [1,1]])
# l = loss(pred, label, mask)*mask.shape[1]/mask.float().sum(dim=1)
# l = nn.functional.binary_cross_entropy_with_logits(pred, target=label,reduction="mean", weight=mask)
# print(l)

# x = torch.sigmoid(pred)
# result = -torch.mean((label*torch.log(x)+(1-label)*torch.log(1-x))*mask)
# print("result:",result)


def sigmd(x):
    return - math.log(1/(1+math.exp(-x)))


embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size))


def train(net, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]   # center.shape=[512, 1]  context_negative.shape=[512, 60]=label.shpae=mask.shape
            pred = skip_gram(center, context_negative, net[0], net[1])   # pred.shape=[512, 1, 60]
            l = (loss(pred.view(label.shape), label, mask)*mask.shape[1]/mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print("epoch:%d, loss:%.2f, time:%.2fs"%(epoch+1, l_sum/n, time.time()-start))


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    cos = torch.matmul(W, x)/(torch.sum(W*W, dim=1)*torch.sum(x*x)+1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:
        print("cosine sim = %.3f: %s"%(cos[i], (idx_to_token[i])))

if __name__ == "__main__":
    train(net, 0.01, 10)
    get_similar_tokens("chip", 3, net[0])

