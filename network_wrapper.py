import torch

from constants import Constants
from network import DeepEvModel


class NetworkWrapper(object):
  def __init__(self, nbets, nactions, ev_embed_dim, bet_embed_dim, device=torch.device("cpu")):
    self._network = DeepEvModel(nbets, nactions, ev_embed_dim=ev_embed_dim, bet_embed_dim=bet_embed_dim).to(device)
    self._network.eval()
    self._device = device
    self._nbets = nbets
    self._nactions = nactions
    self._ev_embed_dim = ev_embed_dim
    self._bet_embed_dim = bet_embed_dim
    print("[NetworkWrapper] Initialized with nbets={} nactions={} ev_embed_dim={} bet_embed_dim={}".format(
        nbets, nactions, ev_embed_dim, bet_embed_dim))

  def network(self):
    return self._network

  def get_action_probabilities(self, infoset, valid_mask):
    """
    Takes an infoset, passes it into the network, and returns the action probabilities predicted
    by the network.
    """
    with torch.no_grad():
      ev_input = infoset.get_ev_input_tensors().to(self._device)

      # Make the opponent bet actions negative, and ours positive.
      bets_input, position_mask = infoset.get_bet_input_tensors()
      
      # Normalize bets_input by the pot size.
      cumul_pot = torch.cumsum(bets_input, dim=1)
      cumul_pot[cumul_pot == 0] = 1
      bets_input = (bets_input * position_mask / cumul_pot).to(self._device)
      # bets_input = (bets_input * position_mask).to(self._device)

      pred_regret = self._network(ev_input, bets_input)[0]

      # Do regret matching on the predicted advantages.
      r_plus = torch.clamp(pred_regret, min=0)

      # As advocated by Brown et. al., choose the action with highest advantage when all of them are
      # negative.
      if r_plus.sum() < 1e-5:
        pred_regret -= pred_regret.min()
        pred_regret *= valid_mask.to(self._device)
        r = torch.zeros_like(r_plus)
        r[torch.argmax(pred_regret)] = 1.0
      else:
        r = r_plus / r_plus.sum()

      return r.cpu()

  def get_action_probabilities_uniform(self):
    return torch.ones(Constants.NUM_ACTIONS) / Constants.NUM_ACTIONS
