import torch

from constants import Constants
from network import DeepCFRModel


class NetworkWrapper(object):
  def __init__(self, ncardtypes, nbets, nactions, embed_dim, device=torch.device("cuda")):
    self._network = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim).to(device)

  def get_action_probabilities(self, infoset):
    """
    Takes an infoset, passes it into the network, and returns the action probabilities predicted
    by the network.
    """
    return self.get_action_probabilities_uniform(infoset)
    # cards_input = 

    # cards (tuple of torch.Tensor): Shape ((B x 2), (B x 3)[, (B x 1), (B x 1)]) # Hole, board [, turn, river]).
    # bets (torch.Tensor) : Shape (batch_size, nbets).

  def get_action_probabilities_uniform(self, infoset):
    return torch.ones(Constants.NUM_ACTIONS) / Constants.NUM_ACTIONS
