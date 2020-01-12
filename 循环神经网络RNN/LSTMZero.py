import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import DataPreprocess as dp
import RNNZero as rnnzero

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dp.load_data_jay_lyrics()

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, dtype=torch.float32,device=device),requires_grad=True))
    W_xi, W_hi, b_i = _three()    # 输入门参数
    W_xf, W_hf, b_f = _three()    # 遗忘门参数
    W_xo, W_ho, b_o = _three()    # 输出门参数
    W_xc, W_hc, b_c = _three()    # 候选记忆细胞参数

    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o,
                             W_xc, W_hc, b_c, W_hq, b_q])

# 自定义模型
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F*C + C_tilda*I
        H = O*torch.tanh(C)
        outputs.append(torch.matmul(H, W_hq) + b_q)
    return outputs, (H, C)


if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]
    rnnzero.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens, vocab_size, device, corpus_indices,
                                  idx_to_char, char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size,
                                  pred_period, pred_len, prefixes)

