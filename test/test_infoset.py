import time, unittest

import torch

from memory_buffer import MemoryBuffer
from infoset import InfoSet, unpack_infoset
from utils import encode_cards_rank_suit


class InfoSetTest(unittest.TestCase):
  def test_info_set_size(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)
    packed = infoset.pack()
    self.assertEqual(len(packed), 1 + 2 + 5 + 24)

  def test_add_cpu(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    max_size = 10000
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cpu"))

    t0 = time.time()
    for i in range(max_size):
      mb.add(infoset, torch.zeros(item_size), 1234)
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
    max_size = 10000
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cuda"))

    t0 = time.time()
    for i in range(max_size):
      mb.add(infoset, torch.zeros(item_size), 1234)
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))

    self.assertTrue(mb.full())

  def test_infoset_pack(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    packed = infoset.pack()

    # First entry is player position (1 in this case).
    self.assertTrue(torch.eq(packed[0], torch.Tensor([1])).all())
    self.assertTrue(torch.eq(packed[1:3], hole_cards).all())
    self.assertTrue(torch.eq(packed[3:6], board_cards).all())

    # -1s to indicate that only the flop exists.
    self.assertTrue(torch.eq(packed[6:8], torch.Tensor([-1, -1])).all())
    self.assertTrue(torch.eq(packed[8:], bet_history_vec).all())

  def test_unpack_infoset(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)
    packed = infoset.pack()

    unpacked = unpack_infoset(packed)
    self.assertTrue((unpacked.bet_history_vec == bet_history_vec).all())
    self.assertTrue((unpacked.board_cards == board_cards).all())
    self.assertTrue((unpacked.hole_cards == hole_cards).all())

  def test_get_card_input_tensors(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    hole_cards_t, board_cards_t = infoset.get_card_input_tensors()
    self.assertEqual(hole_cards_t.shape, (1, 2))
    self.assertEqual(board_cards_t.shape, (1, 5))
    self.assertTrue((board_cards_t[3:] == -1).all())

    print(hole_cards_t)
    print(board_cards_t)

  def test_get_bet_input_tensors(self):
    hole_cards = encode_cards_rank_suit(["Ac", "4s"])
    board_cards = encode_cards_rank_suit(["Js", "Jc", "3h"])
    bet_history_vec = torch.ones(24)
    infoset = InfoSet(hole_cards, board_cards, bet_history_vec, 1)

    bets_t, mask_t = infoset.get_bet_input_tensors()
    self.assertEqual(bets_t.shape, (1, 24))
    self.assertEqual(mask_t.shape, (24,))
    print(bets_t)
    print(mask_t)

if __name__ == "__main__":
  unittest.main()
