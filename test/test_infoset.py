import time, unittest, random

import torch

from memory_buffer import MemoryBuffer
from infoset import EvInfoSet, unpack_ev_infoset, bucket_small, bucket_small_join
from cfr import *
from utils import encode_cards_rank_suit


# class EvInfoSetTest(unittest.TestCase):
#   def test_info_set_size(self):
#     ev = 0.434
#     bet_history_vec = torch.ones(24)
#     infoset = EvInfoSet(ev, bet_history_vec, 1)
#     packed = infoset.pack()
#     self.assertEqual(len(packed), 1 + 1 + 24)
  
#   def test_add_cpu_gpu(self):
#     ev = 0.434
#     bet_history_vec = torch.ones(24)
#     infoset = EvInfoSet(ev, bet_history_vec, 1)

#     info_set_size = 1 + 1 + 24
#     item_size = 32
#     max_size = 10000
#     mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cpu"))

#     # Make buffer on CPU.
#     t0 = time.time()
#     for i in range(max_size):
#       mb.add(infoset, torch.zeros(item_size), 1234)
#     elapsed = time.time() - t0
#     print("Took {} sec".format(elapsed))
#     self.assertTrue(mb.full())

#     # Make buffer on GPU.
#     mb = MemoryBuffer(info_set_size, item_size, max_size=max_size, device=torch.device("cuda"))
#     t0 = time.time()
#     for i in range(max_size):
#       mb.add(infoset, torch.zeros(item_size), 1234)
#     elapsed = time.time() - t0
#     print("Took {} sec".format(elapsed))
#     self.assertTrue(mb.full())
  
#   def test_infoset_pack(self):
#     ev = 0.434
#     bet_history_vec = torch.ones(24)
#     infoset = EvInfoSet(ev, bet_history_vec, 1)
#     packed = infoset.pack()

#     # First entry is player position (1 in this case).
#     self.assertTrue(torch.eq(packed[0], torch.Tensor([1])).all())
#     self.assertTrue(torch.eq(packed[1], ev).all())
#     self.assertTrue(torch.eq(packed[2:], bet_history_vec).all())

#   def test_unpack_infoset(self):
#     ev = 0.434
#     bet_history_vec = torch.ones(24)
#     infoset = EvInfoSet(ev, bet_history_vec, 0)
#     packed = infoset.pack()

#     unpacked = unpack_ev_infoset(packed)
#     self.assertTrue((unpacked.bet_history_vec == bet_history_vec).all())
#     self.assertTrue((unpacked.ev == ev).all())
#     self.assertTrue((unpacked.player_position == 0).all())

#   def test_get_input_tensors(self):
#     ev = 0.434
#     bet_history_vec = torch.ones(24)
#     infoset = EvInfoSet(ev, bet_history_vec, 0)
#     bets_t, mask_t = infoset.get_bet_input_tensors()
#     self.assertEqual(bets_t.shape, (1, 24))
#     self.assertEqual(mask_t.shape, (24,))
#     self.assertEqual(mask_t.sum().item(), 0)
#     print(bets_t)
#     print(mask_t)

#     ev_t = infoset.get_ev_input_tensors()
#     self.assertEqual(ev_t.shape, (1, 1))
#     self.assertAlmostEqual(ev_t.item(), ev)


class BucketTest(unittest.TestCase):
  def test_bucket_small_01(self):
    random.seed(123)
    # P2 is the small blind.
    sb_index = 1
    round_state = create_new_round(sb_index)

    # SB calls, NOT finishing preflop.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    self.assertEqual(bucket[0], 'SB')
    self.assertEqual(bucket[1], 'P')
    self.assertEqual(bucket[2], 'H2')
    self.assertGreaterEqual(infoset.ev, 0.60)
    round_state = round_state.proceed(CallAction())

    # BB checks, ending the preflop.
    infoset = make_infoset(round_state, 0, False)
    bucket = bucket_small(infoset)
    print("SB called preflop:")
    print(bucket)

    round_state = round_state.proceed(CheckAction())

    # BB bets 4.
    infoset = make_infoset(round_state, 0, False)
    # print(round_state.hands)
    # print(round_state.deck.peek(5))
    bucket = bucket_small(infoset)
    print("BB first action of flop:")
    print(bucket)
    self.assertEqual(bucket[0], 'BB')
    self.assertEqual(bucket[1], 'F')
    self.assertEqual(bucket[2], 'H1')
    self.assertGreaterEqual(infoset.ev, 0.40)
    round_state = round_state.proceed(RaiseAction(4))

    # SB raises to 8.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    self.assertEqual(bucket[0], 'SB')
    self.assertEqual(bucket[1], 'F')
    self.assertEqual(bucket[2], 'H1')
    self.assertGreaterEqual(infoset.ev, 0.40)
    self.assertEqual(bucket[7 + 1], 'R') # Opponent did a flop raise.
    self.assertEqual(bucket[11 + 0], '1P') # First action of street was 1P raise.
    round_state = round_state.proceed(RaiseAction(8))

    # BB raises to 12.
    infoset = make_infoset(round_state, 0, False)
    bucket = bucket_small(infoset)
    print(bucket)
    self.assertEqual(bucket[0], 'BB')
    self.assertEqual(bucket[1], 'F')
    self.assertEqual(bucket[2], 'H1')
    self.assertGreaterEqual(infoset.ev, 0.40)
    self.assertEqual(bucket[7 + 1], 'R') # Opponent did a flop raise.
    self.assertEqual(bucket[3 + 1] , 'R') # Player did a flop raise.
    self.assertEqual(bucket[11 + 1], 'HP') # Second action was half pot raise.
    round_state = round_state.proceed(RaiseAction(30))

    # SB calls, ending flop.
    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket)
    self.assertEqual(bucket[0], 'SB')
    self.assertEqual(bucket[1], 'F')
    self.assertEqual(bucket[2], 'H1')
    self.assertGreaterEqual(infoset.ev, 0.40)
    self.assertEqual(bucket[7 + 1], 'R') # Opponent did a flop raise.
    self.assertEqual(bucket[3 + 1] , 'R') # Player did a flop raise.
    self.assertEqual(bucket[11 + 1], 'HP') # Second action was half pot raise.
    self.assertEqual(bucket[11 + 2], '2P') # Third action was a 2pot raise.
    round_state = round_state.proceed(CallAction())

    # Both check on the turn.
    round_state = round_state.proceed(CheckAction())
    round_state = round_state.proceed(CheckAction())

  def test_bucket_small_02(self):
    random.seed(123)
    # P1 is the small blind.
    sb_index = 0
    round_state = create_new_round(sb_index)

    # SB raises to 4.
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(RaiseAction(4))

    # BB calls, ending preflop.
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small(infoset)
    print("Should see opponent raise preflop")
    print(bucket)
    round_state = round_state.proceed(CallAction())

    # BB checks.
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small(infoset)
    print(bucket)
    round_state = round_state.proceed(CheckAction())

    # SB raises to 2.
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small(infoset)
    print("Should see self raise preflop")
    print(bucket)
    round_state = round_state.proceed(RaiseAction(2))

    # BB raises to 40.
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small(infoset)
    print("Should see opponent raise preflop and flop")
    print(bucket)
    round_state = round_state.proceed(RaiseAction(40))

    # SB calls, ending flop.
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small(infoset)
    print("Should see self raise preflop and flop and opponent raise flop")
    print(bucket)
    round_state = round_state.proceed(CallAction())

  def test_bucket_small_03(self):
    # BB.T.H2|x.R.R.x|R.R.x.x|2P.x.x.x': tensor([168., 160., 142., 181., 152., 144.])
    random.seed(123)
    sb_index = 0
    round_state = create_new_round(sb_index)
    round_state = round_state.proceed(RaiseAction(4))   # SB raises.

    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(8))     # BB raises.

    infoset = make_infoset(round_state, 0, True)
    print("SB raises, BB raises")
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(12))  # SB raises.

    infoset = make_infoset(round_state, 1, False)
    print("SB raises, BB raises, SB raises")
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(16))   # BB raises.
    round_state = round_state.proceed(RaiseAction(20))  # SB raises.

    print("SB raise, BB raise, SB raise, BB raise, SB raise")
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(CallAction())     # BB calls.

    infoset = make_infoset(round_state, 1, False)
    print("flop, no actions yet")
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(CheckAction())    # BB checks.

    print("flop, BB checked")
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(CheckAction())    # SB checks.

    print("SB checked, now on turn")
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(10)) # BB raises.
    round_state = round_state.proceed(CallAction())     # SB calls.

    print("turn, BB raised, SB called, now on river")
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(CheckAction())     # BB checks.
    
    print("river, BB checks")
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(10)) # SB raises.

    print("river, BB checks, SB raises")
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(40)) # BB raises.

    print("river, BB checks, SB raises, BB raises")
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(60)) # SB raises.
    round_state = round_state.proceed(RaiseAction(100)) # BB raises.

    print("river, BB check, SB raise, BB raise, SB raise, BB raise")
    infoset = make_infoset(round_state, 0, True)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

    round_state = round_state.proceed(RaiseAction(120)) # SB raises.
    infoset = make_infoset(round_state, 1, False)
    bucket = bucket_small_join(bucket_small(infoset))
    print(bucket)

  def test_exceed_action_limit(self):
    # P2 is the small blind.
    sb_index = 1
    round_state = create_new_round(sb_index)

    # SB calls.
    round_state = round_state.proceed(CallAction())

    # BB checks.
    round_state = round_state.proceed(CheckAction())

    # Do 8 bets/raises to exceed the max 6 actions.
    round_state = round_state.proceed(RaiseAction(2))
    round_state = round_state.proceed(RaiseAction(4))
    round_state = round_state.proceed(RaiseAction(6))
    round_state = round_state.proceed(RaiseAction(8))

    infoset = make_infoset(round_state, 0, False)
    bucket = bucket_small(infoset)
    print(bucket_small_join(bucket))

    round_state = round_state.proceed(RaiseAction(10))

    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket_small_join(bucket))

    round_state = round_state.proceed(RaiseAction(12))
    round_state = round_state.proceed(RaiseAction(14))

    infoset = make_infoset(round_state, 1, True)
    bucket = bucket_small(infoset)
    print(bucket_small_join(bucket))

    round_state = round_state.proceed(RaiseAction(16))
    round_state = round_state.proceed(CallAction())
    
    infoset = make_infoset(round_state, 0, False)
    expected = torch.Tensor([1, 2, 1, 0, 0, 0, 2, 4, 14, 12, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertTrue((infoset.bet_history_vec == expected).all())

    bucket = bucket_small(infoset)
    print(bucket_small_join(bucket))


if __name__ == "__main__":
  unittest.main()
