import torch
from torch import nn


def run_truncated_rnn(model: nn.RNNBase, inp: torch.Tensor, time_dim=1, hidden=None, truncation_conf=None):
    """

    :param model: the rnn or lstm model
    :param inp: input to provide
    :param time_dim: use 1 if lstm is batch_first=True otherwise 0
    :param hidden:
    :param truncation_conf:
    :return:
    """
    assert inp.ndim == 3
    if truncation_conf is None:
        return model(inp, hidden)
    remaining_inp = inp
    idx = [slice(None)] * inp.ndim
    remain_idx = [slice(None)] * inp.ndim
    rnn_out = []
    while True:
        if isinstance(truncation_conf, int):
            next_trunc_idx = min(truncation_conf, remaining_inp.shape[time_dim])
        else:
            next_trunc_idx = torch.randint(
                low=truncation_conf[0], high=truncation_conf[1], size=(1,))[0].clamp(1, remaining_inp.shape[time_dim])
        idx[time_dim] = slice(None, next_trunc_idx)
        remain_idx[time_dim] = slice(next_trunc_idx, None)
        cur_inp = remaining_inp[idx]
        new_rnn_out, hidden = model(cur_inp, hidden)
        for h in hidden:
            h.detach_()
        rnn_out.append(new_rnn_out)
        if next_trunc_idx >= remaining_inp.shape[time_dim]:
            break
        remaining_inp = remaining_inp[remain_idx]
    return torch.cat(rnn_out, dim=time_dim), hidden
