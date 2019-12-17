import unittest
import torch

from network import DeepCFRModel


class NetworkTest(unittest.TestCase):
  def test_forward(self):
    batch_size = 10
    nbets = 2 * 4 * 3       # 3 betting rounds on each street.
    nactions = 20
    embed_dim = 64

    # Test the preflop forward.
    ncardtypes = 1 # Hole only.
    net = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim)

    cards = (torch.zeros(batch_size, 2).long(),)
    bets = torch.zeros(batch_size, nbets)
    action_adv = net.forward(cards, bets)

    # Test the flop forward.
    ncardtypes = 2 # Hole and flop.
    net = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim)

    cards = (
      torch.zeros(batch_size, 2).long(),
      torch.zeros(batch_size, 3).long())
    bets = torch.zeros(batch_size, nbets)
    action_adv = net.forward(cards, bets)

    # Test the turn forward.
    ncardtypes = 3 # Hole, flop, and turn.
    net = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim)

    cards = (
      torch.zeros(batch_size, 2).long(),
      torch.zeros(batch_size, 3).long(),
      torch.zeros(batch_size, 4).long())
    bets = torch.zeros(batch_size, nbets)
    action_adv = net.forward(cards, bets)

    # Test the river forward.
    ncardtypes = 4 # Hole, flop, and turn.
    net = DeepCFRModel(ncardtypes, nbets, nactions, embed_dim)

    cards = (
      torch.zeros(batch_size, 2).long(),
      torch.zeros(batch_size, 3).long(),
      torch.zeros(batch_size, 4).long(),
      torch.zeros(batch_size, 5).long())
    bets = torch.zeros(batch_size, nbets)
    action_adv = net.forward(cards, bets)


if __name__ == "__main__":
  unittest.main()
