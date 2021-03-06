import numpy as np
import torch
from torch import nn, optim
import torch.functional as F
import  DataPreprocess as DP
import RNNZero as RZ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = DP.load_data_jay_lyrics()
num_inputs, num_hiddens, num_ouputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
    W_xz, W_hz, b_z = _three()       # 更新门参数
    W_xr, W_hr, b_r = _three()       # 重置门参数
    W_xh, W_hh, b_h = _three()       # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_ouputs))
    b_q = torch.nn.Parameter(torch.zeros(num_ouputs, device=device, dtype=torch.float32), requires_grad=True)

    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


# 隐藏状态初始化函数
def init_gru_state(bath_size, num_hiddens, device):
    return (torch.zeros((bath_size, num_hiddens), device=device),)


# 定义模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul((R*H), W_hh) + b_h)
        H = Z*H + (1-Z)*H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]
    RZ.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                             char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)







