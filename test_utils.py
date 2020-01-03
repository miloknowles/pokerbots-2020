import torch

from utils import encode_cards_rank_suit
from infoset import InfoSet


def make_dummy_infoset():
  hole_cards = encode_cards_rank_suit(["Ac", "4s"])
  board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
  bet_history_vec = torch.ones(24)
  bet_history_vec[3:7] = 0
  infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)
  return infoset
