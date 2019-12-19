from copy import deepcopy

import torch

from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer
from network_wrapper import NetworkWrapper


def make_infoset(game_state, evt):
  """
  Make an infoset representation for the player about to act.
  """
  # NOTE(milo): Acting position is 0 if this player is the SB (first to act) and 1 if BB.
  small_blind_player_idx = (evt["round_state"]["big_blind_pos"] + 1) % 2
  acting_player_idx = int(evt["round_state"]["next_player"])

  # This is 0 if the current acting player is the SB and 1 if BB.
  acting_player_blind = 0 if small_blind_player_idx == acting_player_idx else 1

  # NOTE(milo): PyPokerEngine encodes cards with rank-suit i.e CJ.
  board_suit_rank = evt["round_state"]["community_card"]

  players = game_state["table"].seats.players
  hole_suit_rank = [
    str(players[acting_player_idx].hole_card[0]),
    str(players[acting_player_idx].hole_card[1])]

  bet_history_vec = torch.zeros(Constants.NUM_BETTING_ACTIONS)
  h = evt["round_state"]["action_histories"]
  
  # Always start out with SB + BB in the pot.
  pot_total = (3 * Constants.SMALL_BLIND_AMOUNT)
  for street in ["preflop", "flop", "turn", "river"]:
    if street in h:
      for i, action in enumerate(h[street]):
        # Percentage of CURRENT pot.
        bet_history_vec[Constants.STREET_OFFSET + i] = action["amount"] / pot_total
        pot_total += action["amount"]

  infoset = InfoSet(
    encode_cards_suit_rank(hole_suit_rank),
    encode_cards_suit_rank(board_suit_rank),
    bet_history_vec,
    acting_player_idx)

  return infoset


def generate_actions(valid_actions, pot_amount):
  """
  Using the valid_actions from the game engine, mask out and scale the entire set of actions.
  """
  actions_mask = torch.zeros(len(Constants.ALL_ACTIONS))
  actions_scaled = deepcopy(Constants.ALL_ACTIONS)

  for item in valid_actions:
    if item["action"] == "fold":
      actions_mask[Constants.ACTION_FOLD] = 1

    elif item["action"] == "call":
      actions_mask[Constants.ACTION_CALL] = 1
      actions_scaled[Constants.ACTION_CALL][1] = item["amount"]

    elif item["action"] == "raise":
      min_raise, max_raise = item["amount"]["min"], item["amount"]["max"]

      actions_mask[Constants.ACTION_MINRAISE] = 1
      actions_mask[Constants.ACTION_MAXRAISE] = 1
      actions_scaled[Constants.ACTION_MINRAISE][1] = min_raise
      actions_scaled[Constants.ACTION_MAXRAISE][1] = max_raise

      if pot_amount <= max_raise:
        actions_mask[Constants.ACTION_POTRAISE] = 1
        actions_scaled[Constants.ACTION_POTRAISE][1] = pot_amount

      if 2 * pot_amount <= max_raise:
        actions_mask[Constants.ACTION_TWOPOTRAISE] = 1
        actions_scaled[Constants.ACTION_TWOPOTRAISE][1] = 2 * pot_amount
      
      if 3 * pot_amount <= max_raise:
        actions_mask[Constants.ACTION_THREEPOTRAISE] = 1
        actions_scaled[Constants.ACTION_THREEPOTRAISE][1] = 3 * pot_amount

  return actions_scaled, actions_mask


if __name__ == "__main__":
  emulator = Emulator()
  emulator.set_game_rule(
    player_num=1,
    max_round=10,
    small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
    ante_amount=0)

  players_info = {}
  players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
  players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}

  initial_state = emulator.generate_initial_game_state(players_info)
  game_state, events = emulator.start_new_round(initial_state)

  p1_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)
  p2_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)

  advantage_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)

  evs = []
  t = 0
  for _ in range(1):
    ev = traverse(game_state, events, emulator, generate_actions, make_infoset,
                  Constants.PLAYER1_UID, p1_strategy, p2_strategy, advantage_mem, strategy_mem, t)
    evs.append(ev)

  avg_ev = np.array(evs).mean()
  print("Average EV={}".format(avg_ev))
