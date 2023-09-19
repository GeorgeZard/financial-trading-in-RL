import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, *args, has_state=False, **kwargs):
        self.has_state = has_state
        super(BaseModel, self).__init__(*args, **kwargs)
