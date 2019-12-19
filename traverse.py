import torch
import numpy as np
from copy import deepcopy

from memory import MemoryBuffer, InfoSet
from utils import *
from constants import Constants


def check_terminal_node(events):
  for e in events:
    if e["type"] == "event_round_finish":
      winning_player = e["winners"][0]["uuid"]
      winning_stack = e["winners"][0]["stack"]
      return True, winning_player, winning_stack
  return False, None, None


def check_player_node(events):
  e = events[-1]
  if e["type"] == "event_ask_player":
    return True, e["uuid"], e
  else:
    return False


def sample_uniform_action(valid_actions):
  item = valid_actions[np.random.randint(len(valid_actions))]
  amount = item["amount"]

  if type(amount) == dict:
    random_amount = np.random.randint(amount["min"], high=amount["max"]+1)
    return item["action"], random_amount
  else:
    return item["action"], item["amount"]


def get_available_actions(valid_actions, pot_amount):
  """
  Using the valid_actions from the game engine, mask out and scale the entire set of actions.
  """
  actions_mask = np.zeros(len(Constants.ALL_ACTIONS))
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
  hole_suit_rank = [str(players[acting_player_idx].hole_card[0]), str(players[acting_player_idx].hole_card[1])]

  bet_history_vec = np.zeros(Constants.NUM_BETTING_ACTIONS)
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


def traverse(game_state, events, emulator, traverse_player, p1_strategy,
             p2_strategy, advantage_mem, strategy_mem, t):
  """
  Recursively traverse the game tree with external sampling.
  """
  # CASE 0: State is a chance node - the game engine takes care of sampling this for us.
  traverse_player_strategy = p1_strategy if traverse_player == "P1" else p2_strategy
  other_player_strategy = p2_strategy if traverse_player == "P1" else p1_strategy

  # CASE 1: State is terminal, return payoff for player.
  is_terminal_node, winning_player, winning_stack = check_terminal_node(events)
  if is_terminal_node:
    payoff = (winning_stack - Constants.INITIAL_STACK)
    return payoff if traverse_player == winning_player else (-1 * payoff)

  # CASE 2: Traversal player action node.
  is_player_node, uuid, evt = check_player_node(events)

  if is_player_node and uuid == traverse_player:
    infoset = make_infoset(game_state, evt)

    pot_size = evt["round_state"]["pot"]["main"]["amount"]
    actions, mask = get_available_actions(evt["valid_actions"], pot_size)

    # Do regret matching to get action probabilities.
    action_probs = traverse_player_strategy.get_action_probabilities(infoset)
    action_probs = apply_mask_and_normalize(action_probs, mask)
    assert np.allclose(action_probs.sum(), 1.0)

    # strategy = np.ones(len(actions)) / len(actions) # Uniform strategy over samples for now.
    values = np.zeros(len(actions))
    instant_regrets = np.zeros(len(actions))

    for i, a in enumerate(actions):
      if mask[i] == 0:
        continue
      updated_state, new_events = emulator.apply_action(game_state, a[0], a[1])
      values[i] = traverse(updated_state, new_events, emulator, traverse_player, p1_strategy,
                           p2_strategy, advantage_mem, strategy_mem, t)
    
    strategy_ev = (action_probs * values).sum()
    for i in range(len(actions)):
      if mask[i] == 0:
        continue
      instant_regrets[i] = (values[i] - strategy_ev)

    # Add the instantaneous regrets to advantage memory for the traversing player.
    # advantage_mem.add_weighted(infoset, instant_regrets, t)

    return strategy_ev

  # CASE 3: Other player action node.
  elif is_player_node and uuid != traverse_player:
    infoset = make_infoset(game_state, evt)

    # External sampling: choose a random action for the non-traversing player.
    pot_size = evt["round_state"]["pot"]["main"]["amount"]
    actions, mask = get_available_actions(evt["valid_actions"], pot_size)

    action_probs = other_player_strategy.get_action_probabilities(infoset)
    action_probs = apply_mask_and_normalize(action_probs, mask)
    assert np.allclose(action_probs.sum(), 1.0)
    action, amount = actions[np.random.choice(len(actions), p=action_probs)]

    # Add the action probabilities to the strategy buffer.
    # strategy_mem.add_weighted(infoset, action_probs, t)

    updated_state, new_events = emulator.apply_action(game_state, action, amount)

    return traverse(updated_state, new_events, emulator, traverse_player, p1_strategy, p2_strategy,
                    advantage_mem, strategy_mem, t)

  else:
    raise NotImplementedError()
