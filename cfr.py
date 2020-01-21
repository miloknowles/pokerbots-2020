import pickle, os

import torch
from constants import Constants

from traverse import create_new_round, make_actions, make_infoset, get_street_0123, make_precomputed_ev
from infoset import bucket_small, bucket_small_join


class RegretMatchedStrategy(object):
  def __init__(self):
    self._regrets = {}

  def size(self):
    return len(self._regrets)

  def add_regret(self, infoset, r):
    """
    Adds an instantaneous regret to total regret.
    """
    assert(len(r) == Constants.NUM_ACTIONS)

    bucket = bucket_small(infoset)
    bstring = bucket_small_join(bucket)

    if bstring not in self._regrets:
      self._regrets[bstring] = torch.zeros(Constants.NUM_ACTIONS)

    self._regrets[bstring] += r

  def get_strategy(self, infoset, valid_mask):
    """
    Does regret matching to return a probabilistic strategy.
    """
    bucket = bucket_small(infoset)
    bstring = bucket_small_join(bucket)

    if bstring not in self._regrets:
      self._regrets[bstring] = torch.zeros(Constants.NUM_ACTIONS)

    total_regret = self._regrets[bstring].clone()

    with torch.no_grad():
      r_plus = torch.clamp(total_regret, min=0)

      # As advocated by Brown et. al., choose the action with highest advantage when all of them are
      # less than zero.
      if r_plus.sum() < 1e-5:
        total_regret -= pred_regret.min()       # Make nonnegative.
        total_regret *= valid_mask              # Mask out illegal actions.
        r = torch.zeros(Constants.NUM_ACTIONS)  # Probability 1 for best action.  
        r[torch.argmax(total_regret)] = 1.0
      else:
        r = r_plus / r_plus.sum()               # Normalize to logit.

      return r.cpu()

  def save(self, filename):
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    with open(filename, "wb") as f:
      pickle.dump(self._regrets, f)
    print("Saved RegretMatchedStrategy to {}".format(filename))

  def load(self, filename):
    with open(filename, "rb") as f:
      self._regrets = pickle.load(f)
    print("Loaded {} items from {}".format(self.size(), filename))


# def traverse_cfr(round_state, traverse_player_idx, sb_player_idx, strategies, advt_mem, strt_mem, t, precomputed_ev, recursion_ctr=[0])
