import unittest, time
from traverse import *
from engine import FoldAction, CallAction, CheckAction, RaiseAction

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

    legal_actions = round_state.legal_actions()
    self.assertTrue(FoldAction in legal_actions)
    self.assertTrue(CallAction in legal_actions)
    self.assertTrue(RaiseAction in legal_actions)

    # SB calls preflop.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1]])

    # BB bets 5 (raises pip to 7)
    round_state = round_state.proceed(RaiseAction(7))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5]])

    # SB raises pip to 10.
    round_state = round_state.proceed(RaiseAction(10))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8]])

    # BB calls, ending the preflop.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], []])

    # BB checks.
    round_state = round_state.proceed(CheckAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0]])

    # SB bets 10, raising pip to 10.
    round_state = round_state.proceed(RaiseAction(10))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10]])

    # BB calls, ending flop.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10, 10], []])

    # BB bets 31, raising pip to 31.
    round_state = round_state.proceed(RaiseAction(31))
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10, 10], [31]])

    # SB calls, ending turn.
    round_state = round_state.proceed(CallAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10, 10], [31, 31], []])

    # Both check on the river.
    round_state = round_state.proceed(CheckAction())
    self.assertEqual(round_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10, 10], [31, 31], [0]])

    terminal_state = round_state.proceed(CheckAction())
    self.assertEqual(terminal_state.previous_state.bet_history, [[1, 2, 1, 5, 8, 3], [0, 10, 10], [31, 31], [0, 0]])


if __name__ == "__main__":
  unittest.main()
