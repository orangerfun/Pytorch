import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import DataPreprocess as DP
import RNNZero

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corpus_indices, char_to_idx, idx_to_char, vocab_size = DP.load_data_jay_lyrics()
print("vocab_size:", vocab_size)

# 定义模型
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, )

# 测试
# num_steps = 3
# batch_size = 2
# state = None
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y, state_new = rnn_layer(X, state)
# print(Y.shape, len(state_new), state_new[0].shape)
# print("====================")
# print("y:",Y)
# print("state:", state_new)

# num_steps = 35
# batch_size = 2
# state = None    # 初始隐藏层状态可以不定义
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y, state_new = rnn_layer(X, state)
# print(Y.shape, len(state_new), state_new.shape)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size*(2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)    # 全连接输出层
        self.state = None

    def forward(self, inputs, state):
        X = RNNZero.to_onehot(inputs, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)  # [num_step*batch_size, num_hiddens]
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, dix_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        X = torch.tensor([output[-1]], device=device).view(1,1)
        if state is not None:
            if isinstance(state, tuple):
                state = state[0].to(device), state[1].to(device)
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return " ".join([idx_to_char[i] for i in output])

# 测试
# model = RNNModel(rnn_layer, vocab_size).to(device)
# print(predict_rnn_pytorch("分开", 10, model, vocab_size, device, idx_to_char, char_to_idx))


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
        char_to_idx, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0., 0, time.time()
        data_iter = DP.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach)
                else:
                    state = state.detach()
            (output, state) = model(X, state)
            y = torch.transpose(Y, 0,1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            RNNZero.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item()*y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum/n)
        except OverflowError:
            perplexity = float("inf")
        if (epoch+1)%pred_period==0:
            print("epoch %d, perplexity %f, time %.2f sec"%(epoch+1, perplexity, time.time()-start))
            for prefix in prefixes:
                print("-",predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))

if __name__ == "__main__":
    model = RNNModel(rnn_layer, vocab_size).to(device)
    num_steps = 32
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ["分开","分开"]
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                        num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)




