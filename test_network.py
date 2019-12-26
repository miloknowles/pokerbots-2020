import unittest
import torch

from network import DeepCFRModel
from network_wrapper import NetworkWrapper
from memory import InfoSet
from utils import *


def make_dummy_infoset():
  hole_cards = encode_cards_rank_suit(["Ac", "4s"])
  board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
  bet_history_vec = torch.ones(24)
  infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)
  return infoset


class NetworkTest(unittest.TestCase):
  def test_forward(self):
    batch_size = 10
    nbets = 2 * 4 * 3       # 3 betting rounds on each street.
    nactions = 4            # 4 output actions at each decision.
    embed_dim = 64

    ncardtypes = 4          # The number of times the card state changes (i.e streets).
    net = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim)

    hole_cards = torch.zeros(batch_size, 2).long()
    bets = torch.zeros(batch_size, nbets)

    for board_card_len in [0, 3, 4, 5]:
      board_cards = -1 * torch.ones(batch_size, 5).long()
      board_cards[:,0:board_card_len] = 0
      board_cards = board_cards.contiguous()
      out = net.forward(hole_cards, board_cards, bets)
      self.assertEqual(out.shape, (batch_size, nactions))


class NetworkWrapperTest(unittest.TestCase):
  def test_get_action_probabilities(self):
    wrap = NetworkWrapper(4, 24, 4, 64, torch.device("cuda:0"))

    infoset = make_dummy_infoset()
    p = wrap.get_action_probabilities(infoset)
    self.assertEqual(p.shape, (4,))


if __name__ == "__main__":
  unittest.main()
