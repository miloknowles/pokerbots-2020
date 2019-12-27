import torch

from constants import Constants
from network import DeepCFRModel


class NetworkWrapper(object):
  def __init__(self, ncardtypes, nbets, nactions, embed_dim, device=torch.device("cpu")):
    self._network = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim).to(device)
    self._network.eval()
    self._device = device
    self._ncardtypes = ncardtypes
    self._nbets = nbets
    self._nactions = nactions
    self._embed_dim = embed_dim
    print("[NetworkWrapper] Initialized with ncardtypes={} nbets={} nactions={} embed_dim={}".format(
      ncardtypes, nbets, nactions, embed_dim))

  def network(self):
    return self._network

  def get_action_probabilities(self, infoset):
    """
    Takes an infoset, passes it into the network, and returns the action probabilities predicted
    by the network.
    """
    with torch.no_grad():
      hole_cards, board_cards = infoset.get_card_input_tensors()
      hole_cards = hole_cards.to(self._device)
      board_cards = board_cards.to(self._device)

      bets_input = infoset.get_bet_input_tensors()[0].to(self._device)

      normalized_adv = self._network(hole_cards, board_cards, bets_input)[0]

      # Do regret matching on the predicted advantages.
      r_plus = torch.clamp(normalized_adv, min=0)

      # As advocated by Brown et. al., choose the action with highest advantage when all of them are
      # negative.
      if r_plus.sum() < 1e-5:
        r = torch.zeros_like(r_plus)
        r[torch.argmax(normalized_adv)] = 1.0
      else:
        r = r_plus / r_plus.sum()

      return r.cpu()

  def get_action_probabilities_uniform(self, infoset):
    return torch.ones(Constants.NUM_ACTIONS) / Constants.NUM_ACTIONS
