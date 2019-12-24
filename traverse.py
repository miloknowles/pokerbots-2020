import torch
# import ray

import numpy as np
from copy import deepcopy
import time

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
    return False, None, None


def check_turn_node(events):
  e = events[-1]
  if e["type"] == "event_ask_player" and e["round_state"]["street"] == "turn":
    return True, e["uuid"], e
  return False, None, None


def traverse(game_state, events, emulator, action_generator, infoset_generator, traverse_player,
             strategies, advantage_mem, strategy_mem, t, recursion_ctr=[0], remote=False,
             external_sampling=True):
  """
  Recursively traverse the game tree with external sampling.
  """
  with torch.no_grad():
    recursion_ctr[0] += 1
    other_player = {Constants.PLAYER1_UID: Constants.PLAYER2_UID,
                    Constants.PLAYER2_UID: Constants.PLAYER1_UID}[traverse_player]

    # CASE 0: State is a chance node - the game engine takes care of sampling this for us.
    is_terminal_node, winning_player, winning_stack = check_terminal_node(events)
    if is_terminal_node:
      payoff = (winning_stack - Constants.INITIAL_STACK)
      return payoff if traverse_player == winning_player else (-1 * payoff)

    # TODO(milo): Remove this!
    is_turn_node, uuid, evt = check_turn_node(events)
    if is_turn_node:
      return 0
    del is_turn_node, uuid, evt

    # CASE 2: Traversal player action node.
    is_player_node, uuid, evt = check_player_node(events)

    if is_player_node and uuid == traverse_player:
      infoset = infoset_generator(game_state, evt)
      pot_size = evt["round_state"]["pot"]["main"]["amount"]
      actions, mask = action_generator(evt["valid_actions"], pot_size)
      del evt, pot_size

      # Do regret matching to get action probabilities.
      action_probs = strategies[traverse_player].get_action_probabilities(infoset)

      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      values = torch.zeros(len(actions))
      instant_regrets = torch.zeros(len(actions))

      for i, a in enumerate(actions):
        if mask[i] == 0:
          continue
        updated_state, new_events = emulator.apply_action(game_state, a[0], a[1]) 
        values[i] = traverse(updated_state, new_events, emulator, action_generator, infoset_generator,
                            traverse_player, strategies, advantage_mem, strategy_mem, t,
                            recursion_ctr=recursion_ctr, remote=remote, external_sampling=external_sampling)
      
      strategy_ev = (action_probs * values).sum().item()
      instant_regrets = (values - strategy_ev * mask)

      # Add the instantaneous regrets to advantage memory for the traversing player.
      if advantage_mem is not None:
        advantage_mem.add(infoset, instant_regrets, t)

      return strategy_ev

    # CASE 3: Other player action node.
    elif is_player_node and uuid != traverse_player:
      infoset = infoset_generator(game_state, evt)

      # External sampling: choose a random action for the non-traversing player.
      pot_size = evt["round_state"]["pot"]["main"]["amount"]
      actions, mask = action_generator(evt["valid_actions"], pot_size)
      del evt, pot_size

      action_probs = strategies[other_player].get_action_probabilities(infoset)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      # Add the action probabilities to the strategy buffer.
      if strategy_mem is not None:
        strategy_mem.add(infoset, action_probs, t)

      # Using external sampling, choose only ONE action for the non-traversal player.
      if external_sampling:
        action, amount = actions[torch.multinomial(action_probs, 1).item()]
        updated_state, new_events = emulator.apply_action(game_state, action, amount)

        # NOTE(milo): Delete all events except the last one to save memory usage.
        return traverse(updated_state, new_events, emulator, action_generator, infoset_generator, traverse_player,
                        strategies, advantage_mem, strategy_mem, t, recursion_ctr=recursion_ctr, remote=remote,
                        external_sampling=external_sampling)
        
      # Otherwise recurse on ALL non-traversal player actions.
      else:
        values = torch.zeros(len(actions))
        for i, a in enumerate(actions):
          if mask[i] == 0:
            continue 
          updated_state, new_events = emulator.apply_action(game_state, a[0], a[1])
          values[i] = traverse(updated_state, new_events, emulator, action_generator, infoset_generator,
                              traverse_player, strategies, advantage_mem, strategy_mem, t,
                              recursion_ctr=recursion_ctr, remote=remote, external_sampling=external_sampling)
        strategy_ev = (action_probs * values).sum().item()
        return strategy_ev

    else:
      raise NotImplementedError()
