import unittest, time
import torch

from network import DeepCFRModel, DeepEvModel
from network_wrapper import NetworkWrapper
from infoset import InfoSet
from utils import *
from test_utils import make_dummy_infoset


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
    valid_mask = torch.ones(4).to(torch.device("cuda:0"))
    p = wrap.get_action_probabilities(infoset, valid_mask)
    self.assertEqual(p.shape, (4,))

  def test_timing(self):
    wrap = NetworkWrapper(4, 24, 4, 64, torch.device("cpu"))

    # Worst case: 1000 hands, 4 streets per hand, avg 4 actions per street
    iters_in_game = 1000 * 4 * 4

    t0 = time.time()
    for _ in range(iters_in_game):
      infoset = make_dummy_infoset()
      valid_mask = torch.ones(4).to(torch.device("cpu"))
      p = wrap.get_action_probabilities(infoset, valid_mask)
    elapsed = time.time() - t0
    print("Time to do {} iters = {} sec".format(iters_in_game, elapsed))


class NewNetworkTest(unittest.TestCase):
  def test_forward(self):
    batch_size = 10
    nbets = 24
    nactions = 4
    embed_dim = 256
    net = DeepEvModel(nbets, nactions, ev_embed_dim=16, bet_embed_dim=embed_dim)

    ev = torch.ones(batch_size, 1)
    bet_features = torch.zeros(batch_size, nbets)
    out = net(ev, bet_features)
  
  def test_timing(self):
    # Worst case: 1000 hands, 4 streets per hand, avg 4 actions per street
    iters_in_game = 1000 * 4 * 4
    nbets = 32
    nactions = 6
    net = DeepEvModel(nbets, 6, ev_embed_dim=16, bet_embed_dim=256)
    torch.set_num_threads(1)

    t0 = time.time()
    for _ in range(iters_in_game):
      out = net(torch.ones(1, 1), torch.ones(1, nbets))
    elapsed = time.time() - t0
    print("Time to do {} iters = {} sec".format(iters_in_game, elapsed))

if __name__ == "__main__":
  unittest.main()
