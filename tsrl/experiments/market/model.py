import torch
from torch import nn, distributions

from tsrl.torch_utils.initialization import init_lstm
from tsrl.torch_utils.model_builder import create_mlp
from tsrl.torch_utils.truncate_bptt import run_truncated_rnn
from tsrl.model import BaseModel
# from dain.dain import DAIN_Layer

from typing import Dict, List, Optional, Tuple

# class AdaptiveBatchNorm1d(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
#         super(AdaptiveBatchNorm1d, self).__init__()
#         self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
#         self.a = nn.Parameter(torch.FloatTensor(1, 1, 1))
#         self.b = nn.Parameter(torch.FloatTensor(1, 1, 1))
#
#     def forward(self, x):
#         print(x.shape)
#         print(self.b * self.bn(x))
#         return self.a * x + self.b * self.bn(x)
#


class MarketAgent(BaseModel):
    """
    The model defined as:
    input_vector -> One LSTM -> FC1 -> Actor-Critic FC (Actor 3 actions (policy), Critic Value function)
    Actor model ->  A Feedforward network 3 with output
    Critic model -> A Feedforward network 1 output

    """
    def __init__(
            self, *, num_inputs,
            lstm_size, critic_size: List[int], actor_size: List[int],
            n_lstm_layers=1, nb_actions=3, dropout=0., pos_encoder_size=10,
            combine_policy_value=True

    ):
        super(MarketAgent, self).__init__(has_state=True)
        self.combine_policy_value = combine_policy_value
        self.num_inputs = num_inputs
        self.policy_lstm = nn.LSTM(num_inputs, lstm_size, num_layers=n_lstm_layers,
                                   batch_first=True)
        init_lstm(self.policy_lstm)
        if not combine_policy_value:
            self.value_lstm = nn.LSTM(num_inputs, lstm_size, num_layers=n_lstm_layers,
                                      batch_first=True)
            init_lstm(self.value_lstm)
        else:
            self.value_lstm = self.policy_lstm
        self.last_states = None
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Linear(3, pos_encoder_size, bias=False)
        self.actor_base = create_mlp(actor_size, nn.SiLU, self.policy_lstm.hidden_size + self.pos_encoder.out_features,
                                     out_size=None, dropout=dropout)
        self.critic_base = create_mlp(critic_size, nn.SiLU,
                                      self.value_lstm.hidden_size + self.pos_encoder.out_features,
                                      out_size=None, dropout=dropout)
        self.actor_head = nn.Linear(actor_size[-1], nb_actions, bias=True)
        self.critic_head = nn.Linear(critic_size[-1], 1, bias=True)
        # self.actor_val_aux_head = create_mlp([256], activations=None, in_size=actor_size[-1],
        # out_size=1,last_lin=True)
        self.actor_val_aux_head = nn.Linear(actor_size[-1], 1, bias=True)
        self.actor_head.weight.data *= 0.1 / self.actor_head.weight.data.norm(dim=1, p=2, keepdim=True)
        self.critic_head.weight.data *= 0.1 / self.critic_head.weight.data.norm(dim=1, p=2, keepdim=True)
        self.actor_val_aux_head.weight.data *= 0.1 / self.actor_val_aux_head.weight.data.norm(dim=1, p=2,
                                                                                              keepdim=True)
        #self.dean = DAIN_Layer(mode='full',  mean_lr=0.00001, gate_lr=0.001, scale_lr=0.0001, input_dim=num_inputs)

    """
    actor-critic deep rl
    """
    def base(self, state=None, calculate_value=True, truncate_bptt=None, **inp) -> Dict[str, torch.Tensor]:
        features, position = inp['features'], inp['position']
        state = {} if state is None else state
        new_state = {}

        # if features.ndim == 3 :
        #     features_dean = self.dean(features.transpose(1,2))
        #     features = features_dean.contiguous().view(features.shape[0], features.shape[1],features.shape[2])
        # else:
        #     features_dean = self.dean(features.reshape(features.shape[0], features.shape[1], 1))
        #     features = features_dean.contiguous().view(features.shape[0], features.shape[1])

        if features.ndim == 2 and position.ndim == 2:
            features = features.view(-1, 1, self.num_inputs)
            position = position.view(-1, 1, position.shape[-1])
        plstm_out, new_state['policy'] = run_truncated_rnn(
            self.policy_lstm, inp=features, hidden=state.get('policy', None), truncation_conf=truncate_bptt)
        enc_pos = self.pos_encoder(position)
        policy_repr = self.actor_base(torch.cat((enc_pos, plstm_out), dim=-1))
        if calculate_value:
            vlstm_out, new_state['value'] = run_truncated_rnn(
                self.value_lstm, inp=features, hidden=state.get('value', None), truncation_conf=truncate_bptt)
            critic_repr = self.critic_base(torch.cat((enc_pos, vlstm_out), dim=-1))
            return dict(policy_repr=policy_repr, critic_repr=critic_repr, state=new_state)
        else:
            return dict(policy_repr=policy_repr, state=new_state)

    def forward(self, state=None, **inp) -> Dict[str, torch.Tensor]:
        base_out = self.base(state=state, calculate_value=False, **inp)
        actor_logits = self.actor_head(base_out['policy_repr'])
        action = actor_logits.argmax(dim=-1).int()
        return dict(market_action=action, state=base_out['state'])

    def sample_eval(self, state=None, **inp) -> Dict[str, torch.Tensor]:
        base_out = self.base(state=state, calculate_value=True, **inp)
        actor_logits = self.actor_head(base_out['policy_repr'])
        dist = distributions.categorical.Categorical(logits=actor_logits)
        value_pred = self.critic_head(base_out['critic_repr'])
        action = dist.sample()
        return dict(pd=dist, value=value_pred,
                    log_prob=dist.log_prob(action),
                    logits=actor_logits, action=action,
                    state=base_out['state'])

    def train_eval(self, features, position, actions,
                   truncate_bptt: Optional[Tuple[int, int]] = None,
                   state=None, policy_aux_val=False):
        base_out = self.base(state=state, features=features,
                             position=position, calculate_value=not policy_aux_val,
                             truncate_bptt=truncate_bptt)
        actor_logits = self.actor_head(base_out['policy_repr'])
        actor_logits = self.dropout(actor_logits)
        dist = distributions.categorical.Categorical(logits=actor_logits)
        res = dict(pd=dist, logits=actor_logits,
                   log_prob=dist.log_prob(actions), state=base_out['state'])
        if policy_aux_val:
            aux_val_pred = self.actor_val_aux_head(self.dropout(base_out['policy_repr']))
            res['aux_val_pred'] = aux_val_pred
        else:
            value_pred = self.critic_head(self.dropout(base_out['critic_repr']))
            res['value'] = value_pred
        return res

    # def step(self, state=None, **inp) -> Dict[str, torch.Tensor]:
    #     features, position = inp['features'], inp['position']
    #     assert features.ndim == 2 and position.ndim == 2
    #     out, new_state = self.policy_lstm(
    #         features.view(-1, 1, self.num_inputs),
    #         state)
    #     out = torch.cat((self.pos_encoder(position.view(-1, 1, position.shape[-1])), out), dim=-1)
    #     actor_out = self.actor(out)
    #     return dict(actor_out=actor_out, state=new_state)
    #
    # def stateful_step(self, state=None, include_value=False, **inp) -> Dict[str, torch.Tensor]:
    #     state = {} if state is None else state
    #     features, position = inp['features'], inp['position']
    #     if features.ndim == 2 and position.ndim == 2:
    #         out, new_policy_state = self.policy_lstm(
    #             features.view(-1, 1, self.num_inputs),
    #             state.get('policy', None))
    #         if not self.combine_policy_value:
    #             value_out, new_value_state = self.value_lstm(
    #                 features.view(-1, 1, self.num_inputs),
    #                 state.get('value', None))
    #         else:
    #             value_out = out
    #         out = torch.cat((self.pos_encoder(position.view(-1, 1, position.shape[-1])),
    #                          out), dim=-1)
    #         value_out = torch.cat((self.pos_encoder(position.view(-1, 1, position.shape[-1])),
    #                                value_out), dim=-1)
    #         # critic_out = self.critic(out)
    #         actor_out = self.actor(out)
    #         actor_dist = distributions.categorical.Categorical(logits=actor_out.squeeze(1))
    #         action = actor_dist.sample()
    #         res = dict(log_prob=actor_dist.log_prob(action).view(-1, 1),
    #                    action=action.view(-1, 1),
    #                    logits=actor_out, pd=actor_dist,
    #                    state=dict(policy=new_policy_state, value=new_value_state),
    #                    )
    #         if include_value:
    #             res['value'] = self.critic(value_out)
    #         return res
    #     else:
    #         raise ValueError(f"Problem with feature shape: {features.shape} "
    #                          f"and position shape: {position.shape}")
    #
    # def train_step(self, features, position, actions,
    #                truncate_bptt: Optional[Tuple[int, int]] = None):
    #     assert features.ndim == 3 and position.ndim == 3
    #
    #     if truncate_bptt is not None:
    #         done = False
    #         states = {}
    #         remaining_features = features
    #         lstm_outs = []
    #         value_lstm_outs = []
    #         while not done:
    #             remaining_ts = remaining_features.shape[1]
    #             next_truncation = \
    #                 torch.randint(low=truncate_bptt[0], high=truncate_bptt[1],
    #                               size=(1,))[0].clamp(1, remaining_ts)
    #             cur_features = remaining_features[:, :next_truncation]
    #             remaining_features = remaining_features[:, next_truncation:]
    #             if next_truncation == remaining_ts:
    #                 done = True
    #             out, states['policy'] = self.policy_lstm(cur_features, states.get('policy', None))
    #             lstm_outs.append(out)
    #             if not self.combine_policy_value:
    #                 value_out, states['value'] = self.value_lstm(cur_features, states.get('value', None))
    #                 value_lstm_outs.append(value_out)
    #             for st in states.values():
    #                 for s in st:
    #                     s.detach_()
    #         lstm_out = torch.cat(lstm_outs, dim=1)
    #         value_lstm_outs = lstm_out if self.combine_policy_value else torch.cat(value_lstm_outs, dim=1)
    #     else:
    #         lstm_out, states = self.policy_lstm(features)
    #         value_lstm_outs = lstm_out if self.combine_policy_value else self.value_lstm(features)[0]
    #     enc_pos = self.pos_encoder(position)
    #     out = torch.cat((enc_pos, lstm_out), dim=-1)
    #     value_out = torch.cat((enc_pos, value_lstm_outs), dim=-1)
    #     critic_out = self.critic(value_out)
    #     actor_out = self.actor(out)
    #     actor_dist = distributions.categorical.Categorical(logits=actor_out)
    #     log_prob = actor_dist.log_prob(actions)
    #     entropy = actor_dist.entropy()
    #     return dict(log_prob=log_prob, value=critic_out,
    #                 logits=actor_out,
    #                 pd=actor_dist, entropy=entropy)

    def weight_stats(self):
        return dict(
            policy_lstm_norm=torch.cat([w.flatten() for w in self.policy_lstm.all_weights[0]],
                                       dim=0).norm().detach().cpu(),
            value_lstm_norm=torch.cat([w.flatten() for w in self.value_lstm.all_weights[0]],
                                      dim=0).norm().detach().cpu(),
            actor_base_norm=self.actor_base[0].weight.norm().detach().cpu(),
            actor_head_norm=self.actor_head.weight.norm().detach().cpu(),
            critic_base_norm=self.critic_base[0].weight.norm().detach().cpu(),
            critic_head_norm=self.critic_head.weight.norm().detach().cpu(),
        )

    """
    For probabilistic knowledge transfer
    """
    def get_features(self, state=None, calculate_value=True, truncate_bptt: Optional[Tuple[int, int]] = None, **inp):
        features, position = inp['features'], inp['position']
        state = {} if state is None else state
        new_state = {}

        if features.ndim == 2 and position.ndim == 2:
            features = features.view(-1, 1, self.num_inputs)
            position = position.view(-1, 1, position.shape[-1])
        plstm_out, new_state['policy'] = run_truncated_rnn(
            self.policy_lstm, inp=features, hidden=state.get('policy', None), truncation_conf=truncate_bptt)
        enc_pos = self.pos_encoder(position)

        return torch.cat((enc_pos, plstm_out), dim=-1)