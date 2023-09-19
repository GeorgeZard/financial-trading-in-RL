import torch
import numpy as np

import warnings


def to_np(t):
    return t.cpu().detach().numpy()


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def set_rng_state(state):
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    else:
        warnings.warn("No numpy random state was provided")
    if 'torch' in state:
        torch.random.set_rng_state(state['torch'])
    else:
        warnings.warn("No torch random state was provide")


def get_rng_state():
    return dict(
        numpy=np.random.get_state(),
        torch=torch.random.get_rng_state()
    )
