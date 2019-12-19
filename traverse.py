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


def traverse(game_state, events, emulator, action_generator, infoset_generator, traverse_player, p1_strategy,
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
    infoset = infoset_generator(game_state, evt)

    pot_size = evt["round_state"]["pot"]["main"]["amount"]
    actions, mask = action_generator(evt["valid_actions"], pot_size)

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
      values[i] = traverse(updated_state, new_events, emulator, action_generator, infoset_generator, traverse_player,
                           p1_strategy, p2_strategy, advantage_mem, strategy_mem, t)
    
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
    infoset = infoset_generator(game_state, evt)

    # External sampling: choose a random action for the non-traversing player.
    pot_size = evt["round_state"]["pot"]["main"]["amount"]
    actions, mask = action_generator(evt["valid_actions"], pot_size)

    action_probs = other_player_strategy.get_action_probabilities(infoset)
    action_probs = apply_mask_and_normalize(action_probs, mask)
    assert np.allclose(action_probs.sum(), 1.0)
    action, amount = actions[np.random.choice(len(actions), p=action_probs)]

    # Add the action probabilities to the strategy buffer.
    # strategy_mem.add_weighted(infoset, action_probs, t)

    updated_state, new_events = emulator.apply_action(game_state, action, amount)

    return traverse(updated_state, new_events, emulator, action_generator, infoset_generator, traverse_player,
                    p1_strategy, p2_strategy, advantage_mem, strategy_mem, t)

  else:
    raise NotImplementedError()
