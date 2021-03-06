import torch
from torch import nn, optim
import DataPreprocess as DP
import RNN as rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = DP.load_data_jay_lyrics()

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)

pred_period, pred_len, prefixes = 40, 50, ["分开", "不分开"]

model = rnn.RNNModel(gru_layer, vocab_size).to(device)

rnn.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
