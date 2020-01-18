import unittest, time
from traverse import *
from engine import FoldAction, CallAction, CheckAction, RaiseAction

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

  def test_make_actions(self):
    pass

if __name__ == "__main__":
  unittest.main()
