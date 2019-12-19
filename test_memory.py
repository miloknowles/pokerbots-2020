import unittest, time
from sys import getsizeof

from memory import InfoSet, MemoryBuffer
from utils import *

import torch


class InfoSetTest(unittest.TestCase):
  def test_info_set(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)
    packed = infoset.pack()
    self.assertEqual(len(packed), 1 + 2 + 5 + 24)

  def test_memory_buffer_size(self):
    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    max_size = int(1e6)
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, store_weights=True)
    print(mb._infosets.dtype)
    print(mb._items.dtype)
    print(mb._weights.dtype)
    print("Memory buffer size (max_size={}): {} mb".format(max_size, mb.size_mb()))

  def test_add_cpu(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    mb = MemoryBuffer(info_set_size, item_size, max_size=int(1e6), store_weights=True, device=torch.device("cpu"))

    t0 = time.time()
    for i in range(int(1e6)):
      mb.add(infoset, torch.zeros(item_size))
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))

    self.assertTrue(mb.full())

  def test_add_gpu(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    mb = MemoryBuffer(info_set_size, item_size, max_size=int(1e6), store_weights=True, device=torch.device("cuda"))

    t0 = time.time()
    for i in range(int(1e6)):
      mb.add(infoset, torch.zeros(item_size))
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))

    self.assertTrue(mb.full())


if __name__ == "__main__":
  unittest.main()
