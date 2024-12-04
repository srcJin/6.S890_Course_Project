import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    # def forward(self, agent_qs, states):
    #     bs = agent_qs.size(0)
    #     states = states.reshape(-1, self.state_dim)
    #     agent_qs = agent_qs.view(-1, 1, self.n_agents)
    #     # First layer
    #     w1 = th.abs(self.hyper_w_1(states))
    #     b1 = self.hyper_b_1(states)
    #     w1 = w1.view(-1, self.n_agents, self.embed_dim)
    #     b1 = b1.view(-1, 1, self.embed_dim)
    #     hidden = F.elu(th.bmm(agent_qs, w1) + b1)
    #     # Second layer
    #     w_final = th.abs(self.hyper_w_final(states))
    #     w_final = w_final.view(-1, self.embed_dim, 1)
    #     # State-dependent bias
    #     v = self.V(states).view(-1, 1, 1)
    #     # Compute final output
    #     y = th.bmm(hidden, w_final) + v
    #     # Reshape and return
    #     q_tot = y.view(bs, -1, 1)
    #     return q_tot

    def forward(self, agent_qs, states):
        # Log input dimensions and values
        logger.debug(f"qmix.py Forward pass called with agent_qs shape: {agent_qs.shape}")
        logger.debug(f"qmix.py Forward pass called with states shape: {states.shape}")

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # Log reshaped inputs
        logger.debug(f"qmix.py Reshaped states to: {states.shape}")
        logger.debug(f"qmix.py Reshaped agent_qs to: {agent_qs.shape}")

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        # Log weights and biases for the first layer
        logger.debug(f"qmix.py w1 shape: {w1.shape}, b1 shape: {b1.shape}")
        logger.debug(f"qmix.py w1 sample: {w1[0]}")
        logger.debug(f"qmix.py b1 sample: {b1[0]}")

        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Log hidden layer output
        logger.debug(f"qmix.py Hidden layer output shape: {hidden.shape}")
        logger.debug(f"qmix.py Hidden layer output sample: {hidden[0]}")

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # Log weights for the second layer
        logger.debug(f"w_final shape: {w_final.shape}")
        logger.debug(f"w_final sample: {w_final[0]}")

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Log state-dependent bias
        logger.debug(f"v shape: {v.shape}")
        logger.debug(f"v sample: {v[0]}")

        # Compute final output
        y = th.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        # Log final output
        logger.debug(f"Final q_tot shape: {q_tot.shape}")
        logger.debug(f"Final q_tot sample: {q_tot[0]}")

        return q_tot