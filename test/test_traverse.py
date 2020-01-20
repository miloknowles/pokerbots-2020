import unittest, time, random
from traverse import *
from network_wrapper import NetworkWrapper
from engine import FoldAction, CallAction, CheckAction, RaiseAction
from pbots_calc import CalcWithLookup

import torch


class TraverseTest(unittest.TestCase):
  def test_create_new_round(self):
    for sb_index in (0, 1):
      round_state = create_new_round(sb_index)

      # P1 should have the first action.
      self.assertEqual(round_state.button, sb_index)
      self.assertEqual(round_state.street, 0)

      self.assertEqual(round_state.pips[sb_index], 1)
      self.assertEqual(round_state.pips[1 - sb_index], 2)
      self.assertEqual(round_state.stacks[sb_index], 199)
      self.assertEqual(round_state.stacks[1 - sb_index], 198)

  def test_proceed(self):
    # P1 is the small blind.
    sb_index = 0
    round_state = create_new_round(sb_index)
    actions, mask = make_actions(round_state)
    self.assertEqual(actions, [FoldAction(), CallAction(), CheckAction(),
        RaiseAction(amount=4), RaiseAction(amount=6), RaiseAction(amount=10)])
    self.assertTrue((mask == torch.Tensor([1, 1, 0, 1, 1, 1])).all())

    # SB calls preflop.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1]])

    # BB bets 6 (raises pip to 8)
    actions, mask = make_actions(round_state)
    self.assertEqual(actions, [FoldAction(), CallAction(), CheckAction(),
        RaiseAction(amount=4), RaiseAction(amount=6), RaiseAction(amount=10)])
    self.assertTrue((mask == torch.Tensor([0, 0, 1, 1, 1, 1])).all())
    round_state = round_state.proceed(RaiseAction(8))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6]])

    # SB raises pip to 14 (adds 6 on top of call).
    actions, mask = make_actions(round_state)
    self.assertEqual(actions, [FoldAction(), CallAction(), CheckAction(),
        RaiseAction(amount=16), RaiseAction(amount=24), RaiseAction(amount=40)])
    self.assertTrue((mask == torch.Tensor([1, 1, 0, 1, 1, 1])).all())
    round_state = round_state.proceed(RaiseAction(14))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12]])

    # BB calls, ending the preflop.
    actions, mask = make_actions(round_state)
    self.assertEqual(actions, [FoldAction(), CallAction(), CheckAction(),
        RaiseAction(amount=28), RaiseAction(amount=42), RaiseAction(amount=70)])
    self.assertTrue((mask == torch.Tensor([1, 1, 0, 1, 1, 1])).all())
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], []])

    # BB checks.
    actions, mask = make_actions(round_state)
    self.assertEqual(actions, [FoldAction(), CallAction(), CheckAction(),
        RaiseAction(amount=14), RaiseAction(amount=28), RaiseAction(amount=56)])
    round_state = round_state.proceed(CheckAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0]])

    # SB bets 10, raising pip to 10.
    round_state = round_state.proceed(RaiseAction(10))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10]])

    # BB calls, ending flop.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10, 10], []])

    # BB bets 31, raising pip to 31.
    round_state = round_state.proceed(RaiseAction(31))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10, 10], [31]])

    # SB calls, ending turn.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10, 10], [31, 31], []])

    # Both check on the river.
    round_state = round_state.proceed(CheckAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10, 10], [31, 31], [0]])

    terminal_state = round_state.proceed(CheckAction())
    self.assertEqual(terminal_state.previous_state.bet_history, [[1, 2, 1, 6, 12, 6], [0, 10, 10], [31, 31], [0, 0]])

    infoset = make_infoset(round_state, 0, True)
    self.assertEqual(infoset.player_position, 0)
    expected_history = torch.Tensor([1, 2, 1, 6, 12, 6, 0, 10, 10, 0, 0, 0, 31, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertTrue((expected_history == infoset.bet_history_vec).all())
    print(infoset.bet_history_vec)

  def test_bet_history_wraps(self):
    # P2 is the small blind.
    sb_index = 1
    round_state = create_new_round(sb_index)

    # SB calls, BB checks.
    round_state = round_state.proceed(CallAction())

    # Do 8 bets/raises to exceed the max 6 actions.
    round_state = round_state.proceed(RaiseAction(2))
    round_state = round_state.proceed(RaiseAction(4))
    round_state = round_state.proceed(RaiseAction(6))
    round_state = round_state.proceed(RaiseAction(8))
    round_state = round_state.proceed(RaiseAction(10))
    round_state = round_state.proceed(RaiseAction(12))
    round_state = round_state.proceed(RaiseAction(14))
    round_state = round_state.proceed(RaiseAction(16))
    round_state = round_state.proceed(CallAction())
    
    infoset = make_infoset(round_state, 0, False)
    expected = torch.Tensor([1, 2, 1, 0, 0, 0, 2, 4, 4, 4, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertTrue((infoset.bet_history_vec == expected).all())
    print(infoset.bet_history_vec)

  def test_traverse_timing(self):
    t = 1
    sb_index = 0
    traverse_player_idx = 0

    strategies = [
      NetworkWrapper(Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 8, 64, torch.device("cpu")),
      NetworkWrapper(Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 8, 64, torch.device("cpu"))
    ]

    profiler = {"create_new_round": 0, "make_precomputed_ev": 0, "traverse": 0}

    tstart = time.time()
    N = 100
    for _ in range(N):
      ctr = [0]
      t0 = time.time()
      round_state = create_new_round(sb_index)
      elapsed = time.time() - t0
      profiler["create_new_round"] += elapsed
      t0 = time.time()
      precomputed_ev = make_precomputed_ev(round_state)
      profiler["make_precomputed_ev"] += elapsed
      elapsed = time.time() - t0
      t0 = time.time()
      info = traverse(round_state, make_actions, make_infoset, traverse_player_idx, sb_index,
                      strategies, None, None, t, precomputed_ev, recursion_ctr=ctr)
      elapsed = time.time() - t0
      profiler["traverse"] += elapsed
      print(ctr)

    elapsed = time.time() - tstart
    print(profiler)
    print("Took {} sec for {} traversals".format(elapsed, N))

  def test_ev_variance(self):
    calculator = CalcWithLookup()

    results = []
    # Flop results:
    # stdev is ~0.05 w/ 100 MC iters
    # stdev is ~0.035 w/ 200 MC iters.
    # stdev is ~0.02 w/ 500 MC iters.
    # stdev is ~0.015 w/ 1000 MC iters.
    for _ in range(1000):
      results.append(calculator.calc(["3h", "Th"], b"3c7h7d6c8d", b"", 1326))

    variance = np.var(results)
    stdev = np.sqrt(variance)
    print("stdev={}".format(stdev))

  def test_traverse_deterministic(self):
    t = 0
    sb_index = 0
    traverse_player_idx = 0

    random.seed(1234)
    round_state = create_new_round(sb_index)
    print(round_state.hands)
    print(round_state.deck.peek(5))

    strategies = [
      NetworkWrapper(Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 8, 64, torch.device("cuda:0")),
      NetworkWrapper(Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 8, 64, torch.device("cuda:0"))
    ]

    ctr = [0]
    precomputed_ev = make_precomputed_ev(round_state)
    info = traverse(round_state, make_actions, make_infoset, traverse_player_idx, sb_index,
                    strategies, None, None, t, precomputed_ev, recursion_ctr=ctr)
    print(info.exploitability)
    print(info.strategy_ev)
    print(info.best_response_ev)

if __name__ == "__main__":
  unittest.main()
