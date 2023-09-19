from torch import nn


def init_lstm(lstm_module):
    for n, w in lstm_module.named_parameters():
        if "weight_ih" in n:
            nn.init.xavier_normal_(w)
        elif "weight_hh" in n:
            lstm_size = w.size(0) // 4
            for i in range(0, w.size(0), lstm_size):
                nn.init.orthogonal_(w[i: i + lstm_size])
        elif "bias_hh" in n:
            w.data[w.size(0) // 4: w.size(0) // 2].fill_(1.0)
