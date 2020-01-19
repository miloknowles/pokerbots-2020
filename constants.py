from engine import FoldAction, CallAction, CheckAction, RaiseAction


class Constants(object):
  PLAYER1_UID = "P1"
  PLAYER2_UID = "P2"
  INITIAL_STACK = 200
  SMALL_BLIND_AMOUNT = 1
  ANTE_AMOUNT = 0

  BET_ACTIONS_PER_STREET = 4
  BET_HISTORY_SIZE = 16 # 4 streets, 6 betting actions per street.

  # INFO_SET_SIZE = 1 + 2 + 5 + BET_HISTORY_SIZE
  INFO_SET_SIZE = 1 + BET_HISTORY_SIZE # EV and betting actions.

  ALL_ACTIONS = [
    FoldAction(),
    CallAction(),
    CheckAction(),
    RaiseAction(amount=0.5),
    RaiseAction(amount=1),
    RaiseAction(amount=2),
    # RaiseAction(amount=3)
  ]

  NUM_ACTIONS = len(ALL_ACTIONS)

  # # Index of these actions in ALL_ACTIONS.
  # ACTION_FOLD = 0
  # ACTION_CALL = 1
  # ACTION_POTRAISE = 2
  # ACTION_TWOPOTRAISE = 3
  # ACTION_THREEPOTRAISE = 4

  NUM_STREETS = 4
