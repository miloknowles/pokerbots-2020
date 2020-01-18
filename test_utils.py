import torch

from utils import encode_cards_rank_suit
from infoset import EvInfoSet


def make_dummy_ev_infoset():
  ev = 0.43
  bet_history_vec = torch.ones(24)
  bet_history_vec[3:7] = 0
  infoset = EvInfoSet(ev, bet_history_vec, 1)
  return infoset
