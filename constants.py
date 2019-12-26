class Constants(object):
  PLAYER1_UID = "P1"
  PLAYER2_UID = "P2"
  INITIAL_STACK = 100
  SMALL_BLIND_AMOUNT = 1
  ANTE_AMOUNT = 0

  STREET_OFFSET = 6
  NUM_BETTING_ACTIONS = 24 # 4 streets, 6 betting actions per street.

  INFO_SET_SIZE = 1 + 2 + 5 + NUM_BETTING_ACTIONS

  ALL_ACTIONS = [
    ["fold", 0],
    ["call", -1],
    # ["raise", -1],
    ["raise", -1],
    # ["raise", -1],
    # ["raise", -1],
    ["raise", -1]
  ]

  NUM_ACTIONS = len(ALL_ACTIONS)

  ACTION_FOLD = 0
  ACTION_CALL = 1
  # ACTION_MINRAISE = 2
  ACTION_POTRAISE = 2
  # ACTION_TWOPOTRAISE = 4
  # ACTION_THREEPOTRAISE = 5
  ACTION_MAXRAISE = 3

  NUM_STREETS = 4
