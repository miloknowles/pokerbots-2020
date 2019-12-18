import unittest, time
from sys import getsizeof
from memory import InfoSet, MemoryBuffer
from utils import *


class InfoSetTest(unittest.TestCase):
  def test_info_set(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = np.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec)
    packed = infoset.pack()
    self.assertEqual(len(packed), 2 + 5 + 24)

  def test_memory_buffer_size(self):
    info_set_size = 2 + 5 + 24
    item_size = 64
    mb = MemoryBuffer(info_set_size, item_size, max_size=80000, store_weights=True)
    print(mb._infosets.dtype)
    print(mb._items.dtype)
    print(mb._weights.dtype)
    print("Memory buffer size: {} Mb".format(getsizeof(mb)))

  def test_add(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = np.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec)
    packed = infoset.pack()

    info_set_size = 2 + 5 + 24
    item_size = 64
    mb = MemoryBuffer(info_set_size, item_size, max_size=80000, store_weights=True)

    t0 = time.time()
    for i in range(80000):
      mb.add(packed, np.zeros(item_size))
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))

    self.assertTrue(mb.full())


if __name__ == "__main__":
  unittest.main()
