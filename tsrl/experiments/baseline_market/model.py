import torch
from torch import nn, distributions

from tsrl.torch_utils.initialization import init_lstm
from tsrl.torch_utils.model_builder import create_mlp
from tsrl.torch_utils.truncate_bptt import run_truncated_rnn
from tsrl.model import BaseModel

from typing import Dict, List, Optional, Tuple


class BaseNet(BaseModel):
    def __init__(self, num_inputs, lstm_size, mlp_size, out_size=1):
        super(BaseNet, self).__init__(has_state=True)
        self.lstm = nn.LSTM(num_inputs, lstm_size, batch_first=True)
        self.mlp = create_mlp(layer_sizes=mlp_size, activations=nn.PReLU,
                              in_size=self.lstm.hidden_size, out_size=out_size)

    def forward(self, x, state=None):
        if x.ndim == 2:
            x = x.view(-1, 1, self.num_inputs)
        lstm_out = self.lstm(x, state=state)
        out = self.mlp(lstm_out)
        return out


class BaselineSAC(BaseModel):
    def __init__(self, num_inputs, lstm_size, mlp_size):
        super(BaselineSAC, self).__init__(has_state=True)
        modelargs = num_inputs, lstm_size, mlp_size
        self.policy = BaseNet(*modelargs, out_size=3)
        self.critic = BaseNet(*modelargs, out_size=1)
