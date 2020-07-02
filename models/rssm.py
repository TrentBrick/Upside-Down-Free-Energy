# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from .decoder import ConvDecoder
from .encoder import ConvEncoder
from .dynamics import RecurrentDynamics 
from .rewards import RewardModel

class RSSModel(object):
    def __init__(
        self,
        action_size,
        hidden_size,
        state_size,
        embedding_size,
        node_size,
        decoder_reward_condition,
        decoder_make_sigmas,
        device="cpu",
    ):

        self.action_size = action_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.node_size = node_size
        self.decoder_reward_condition = decoder_reward_condition
        self.decoder_make_sigmas = decoder_make_sigmas
        self.device = device

        self.encoder = ConvEncoder(embedding_size).to(device)
        self.decoder = ConvDecoder(hidden_size, state_size, embedding_size, self.decoder_reward_condition, self.decoder_make_sigmas).to(device)
        self.reward_model = RewardModel(hidden_size, state_size, node_size).to(device)

        self.dynamics = RecurrentDynamics(
            hidden_size, state_size, action_size, node_size, embedding_size
        ).to(device)

    def parameters(self):
        return (
            list(self.decoder.parameters())
            + list(self.encoder.parameters())
            + list(self.reward_model.parameters())
            + list(self.dynamics.parameters())
        )

    def perform_rollout(
        self, actions, hidden=None, state=None, encoder_output=None, non_terms=None
    ):
        if hidden is not None and state is not None:
            """ [action] (seq_len, batch_size, action_size )
                [hidden] (batch_size, hidden_size ) 
                [state]  (batch_size, state_size ) 
            """
            return self.dynamics(hidden, state, actions, encoder_output, non_terms)
        else:
            """ [action] (seq_len, batch_size, n_actions ) 
                [hidden] (seq_len, batch_size, hidden_size ) 
                [state]  (seq_len, batch_size, state_size ) 
            """

            batch_size = encoder_output.size(1)
            init_hidden, init_state = self.init_hidden_state(batch_size)

            return self.dynamics(
                init_hidden, init_state, actions, encoder_output=encoder_output, non_terms=non_terms
            )

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.dynamics.eval()
        self.reward_model.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.dynamics.train()
        self.reward_model.train()

    def encode_obs(self, obs):
        return self.encoder(obs)

    def encode_sequence_obs(self, obs):
        return self._bottle(self.encoder, (obs,))

    def decode_obs(self, hiddens, posterior_states, rewards=None):
        return self.decoder(hiddens, posterior_states, rewards)

    def decode_sequence_obs(self, hiddens, posterior_states, rewards=None):
        return self._bottle(self.decoder, (hiddens, posterior_states, rewards))

    def decode_reward(self, hiddens, posterior_states):
        return self.reward_model(hiddens, posterior_states)

    def decode_sequence_reward(self, hiddens, posterior_states):
        return self._bottle(self.reward_model, (hiddens, posterior_states))

    def init_hidden_state(self, batch_size):
        init_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        init_state = torch.zeros(batch_size, self.state_size).to(self.device)
        return init_hidden, init_state

    def init_hidden_state_action(self, batch_size):
        init_hidden, init_state = self.init_hidden_state(batch_size)
        action = torch.zeros(batch_size, self.action_size).to(self.device)
        return init_hidden, init_state, action

    def get_save_dict(self):
        return {
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict(),
        }

    def load_state_dict(self, model_dict):
        self.dynamics.load_state_dict(model_dict["dynamics"])
        self.encoder.load_state_dict(model_dict["encoder"])
        self.decoder.load_state_dict(model_dict["decoder"])
        self.reward_model.load_state_dict(model_dict["reward_model"])

    def _bottle(self, f, x_tuple):
        """ Flattens the first two dimensions (batch and sequence len) before 
        applying the desired function. Returns as the original size. 
        loops over the first dims of x [seq_len] and applies f """
        # recording the sizes of everything passed in. 
        x_sizes = tuple(map(lambda x: x.size(), x_tuple))
        # applies the transform.
        y = f(
            # x here is a tuple of (object, its sizes)
            *map(
                #combine the batch and sequence dimensions. respects all others. 
                lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)
            )
        )
        if type(y)==tuple and len(y)==2:
            # should work for anything where the NN has multiple outputs. 
            val1, val2 = y[0], y[1]
            return val1.view(x_sizes[0][0], x_sizes[0][1], *val1.size()[1:]), \
                val2.view(x_sizes[0][0], x_sizes[0][1], *val2.size()[1:])

        else: 
            y_size = y.size()
            # respecting the batch and seq dimensions
            return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
