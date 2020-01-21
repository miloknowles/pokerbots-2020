import time, unittest, random

import torch

from memory_buffer import MemoryBuffer
from infoset import EvInfoSet, unpack_ev_infoset, bucket_small
from traverse import *
from utils import encode_cards_rank_suit


class EvInfoSetTest(unittest.TestCase):
  def test_info_set_size(self):
    ev = 0.434
    bet_history_vec = torch.ones(24)
    infoset = EvInfoSet(ev, bet_history_vec, 1)
    packed = infoset.pack()
    self.assertEqual(len(packed), 1 + 1 + 24)
  
  def test_add_cpu_gpu(self):
    ev = 0.434
    bet_history_vec = torch.ones(24)
    infoset = EvInfoSet(ev, bet_history_vec, 1)

    info_set_size = 1 + 1 + 24
    item_size = 32
    max_size = 10000
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cpu"))

    # Make buffer on CPU.
    t0 = time.time()
    for i in range(max_size):
      mb.add(infoset, torch.zeros(item_size), 1234)
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))
    self.assertTrue(mb.full())

    # Make buffer on GPU.
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cuda"))
    t0 = time.time()
    for i in range(max_size):
      mb.add(infoset, torch.zeros(item_size), 1234)
    elapsed = time.time() - t0
    print("Took {} sec".format(elapsed))
    self.assertTrue(mb.full())
  
  def test_infoset_pack(self):
    ev = 0.434
    bet_history_vec = torch.ones(24)
    infoset = EvInfoSet(ev, bet_history_vec, 1)
    packed = infoset.pack()

    # First entry is player position (1 in this case).
    self.assertTrue(torch.eq(packed[0], torch.Tensor([1])).all())
    self.assertTrue(torch.eq(packed[1], ev).all())
    self.assertTrue(torch.eq(packed[2:], bet_history_vec).all())

  def test_unpack_infoset(self):
    ev = 0.434
    bet_history_vec = torch.ones(24)
    infoset = EvInfoSet(ev, bet_history_vec, 0)
    packed = infoset.pack()

    unpacked = unpack_ev_infoset(packed)
    self.assertTrue((unpacked.bet_history_vec == bet_history_vec).all())
    self.assertTrue((unpacked.ev == ev).all())
    self.assertTrue((unpacked.player_position == 0).all())

  def test_get_input_tensors(self):
    ev = 0.434
    bet_history_vec = torch.ones(24)
    infoset = EvInfoSet(ev, bet_history_vec, 0)
    bets_t, mask_t = infoset.get_bet_input_tensors()
    self.assertEqual(bets_t.shape, (1, 24))
    self.assertEqual(mask_t.shape, (24,))
    self.assertEqual(mask_t.sum().item(), 0)
    print(bets_t)
    print(mask_t)

    ev_t = infoset.get_ev_input_tensors()
    self.assertEqual(ev_t.shape, (1, 1))
    self.assertAlmostEqual(ev_t.item(), ev)


class BucketTest(unittest.TestCase):
  def test_bucket_small(self):
    random.seed(123)
    # P2 is the small blind.
    sb_index = 1
    round_state = create_new_round(sb_index)

    # SB calls, finishing preflop.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(CallAction())

    # BB bets 4.
    infoset = make_infoset(round_state, 0, False)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(RaiseAction(4))

    # SB raises to 8.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(RaiseAction(8))

    # BB raises to 12.
    infoset = make_infoset(round_state, 0, False)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(RaiseAction(30))

    # SB calls, ending flop.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(CallAction())

    # Both check on the turn.
    round_state = round_state.proceed(CheckAction())
    round_state = round_state.proceed(CheckAction())


if __name__ == "__main__":
  unittest.main()
