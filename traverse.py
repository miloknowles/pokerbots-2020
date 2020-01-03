import ray
import torch

import numpy as np
from copy import deepcopy
import time

from utils import *
from constants import Constants
from utils import sample_uniform_action


class TraverseMode(object):
  TRAVERSE_PRE_TURN = 0
  TRAVERSE_POST_TURN = 1
  TRAVERSE_FULL = 2


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


def traverse_until_turn(game_state, events, emulator, action_generator, traverse_player,
                        recursion_ctr=[0]):
  """
  Randomly sample actions to traverse until a turn action node is reached. Returns the game_state
  and new_events at the first turn node that is reached.
  """
  with torch.no_grad():
    recursion_ctr[0] += 1
    other_player = {Constants.PLAYER1_UID: Constants.PLAYER2_UID,
                    Constants.PLAYER2_UID: Constants.PLAYER1_UID}[traverse_player]

    # Base case: first action of the turn, return the state and event.
    is_turn_node, uuid, evt = check_turn_node(events)
    if is_turn_node:
      return (game_state, evt)

    is_player_node, uuid, evt = check_player_node(events)

    if is_player_node:
      action, amount = sample_uniform_action(evt["valid_actions"])
      updated_state, new_events = emulator.apply_action(game_state, action, amount)

      return traverse_until_turn(updated_state, new_events, emulator, action_generator,
                      traverse_player, recursion_ctr=recursion_ctr)
    else:
      return None


class TreeNodeInfo(object):
  @staticmethod
  def uuid_to_index(uuid):
    return 0 if uuid == Constants.PLAYER1_UID else 1

  def __init__(self):
    # Expected value for each player at this node if they play according to their current strategy.
    self.strategy_ev = torch.zeros(2)

    # Expected value for each player if they choose a best-response strategy given the other.
    self.best_response_ev = torch.zeros(2)

    # The difference in EV between the best response strategy and the current strategy.
    self.exploitability = torch.zeros(2)


def traverse(game_state, events, emulator, action_generator, infoset_generator,
             traverse_player, strategies, advt_mem, strt_mem, t, recursion_ctr=[0],
             do_external_sampling=True):
  """
  Recursively traverse the game tree with external sampling.

  Returns:
    (TreeNodeInfo)
  """
  with torch.no_grad():
    node_info = TreeNodeInfo()
    tp_index = TreeNodeInfo.uuid_to_index(traverse_player)

    recursion_ctr[0] += 1
    other_player = {Constants.PLAYER1_UID: Constants.PLAYER2_UID,
                    Constants.PLAYER2_UID: Constants.PLAYER1_UID}[traverse_player]

    # CASE 0: State is a chance node - the game engine takes care of sampling this for us.
    is_terminal_node, winning_player, winning_stack = check_terminal_node(events)
    if is_terminal_node:
      payoff = (winning_stack - Constants.INITIAL_STACK)
      node_info.strategy_ev[0] = payoff if winning_player == Constants.PLAYER1_UID else -1 * payoff
      node_info.strategy_ev[1] = -1 * node_info.strategy_ev[0] # Zero sum.
      node_info.best_response_ev = node_info.strategy_ev
      return node_info

    # CASE 1: Traversal player action node.
    is_player_node, uuid, evt = check_player_node(events)

    if is_player_node and uuid == traverse_player:
      infoset = infoset_generator(game_state, evt)
      pot_size = evt["round_state"]["pot"]["main"]["amount"]
      actions, mask = action_generator(evt["valid_actions"], pot_size)
      del evt, pot_size

      # Do regret matching to get action probabilities.
      if t == 0:
        action_probs = strategies[traverse_player].get_action_probabilities_uniform()
      else:
        action_probs = strategies[traverse_player].get_action_probabilities(infoset, mask)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      action_values = torch.zeros(2, len(actions))
      br_values = torch.zeros(2, len(actions))
      instant_regrets = torch.zeros(len(actions))

      plyr_idx = TreeNodeInfo.uuid_to_index(uuid)
      opp_idx = (1 - plyr_idx)

      for i, a in enumerate(actions):
        if mask[i] == 0:
          continue
        updated_state, new_events = emulator.apply_action(game_state, a[0], a[1]) 
        child_node_info = traverse(
            updated_state, new_events, emulator, action_generator, infoset_generator,
            traverse_player, strategies, advt_mem, strt_mem, t,
            recursion_ctr=recursion_ctr, do_external_sampling=do_external_sampling)
        
        # Expected value of the acting player taking this action and then continuing according to their strategy.
        action_values[:,i] = child_node_info.strategy_ev

        # Expected value for each player if the acting player takes this action and then they both
        # follow a best-response strategy.
        br_values[:,i] = child_node_info.best_response_ev
      
      # Sum along every action multiplied by its probability of occurring.
      node_info.strategy_ev = (action_values * action_probs).sum(axis=1)

      # Compute the instantaneous regrets for the traversing player.
      instant_regrets_tp = (action_values[tp_index] - (node_info.strategy_ev[tp_index] * mask))

      # The acting player chooses the BEST action with probability 1, while the opponent best
      # response EV depends on the reach probability of their next acting situation.
      node_info.best_response_ev[plyr_idx] = torch.max(br_values[plyr_idx,:])
      node_info.best_response_ev[opp_idx] = torch.sum(action_probs * br_values[opp_idx,:])

      # Exploitability is the difference in payoff between a local best response strategy and the
      # full mixed strategy.
      node_info.exploitability = node_info.best_response_ev - node_info.strategy_ev

      # Add the instantaneous regrets to advantage memory for the traversing player.
      if advt_mem is not None:
        advt_mem.add(infoset, instant_regrets_tp, t)

      return node_info

    # CASE 2: Other player action node.
    elif is_player_node and uuid != traverse_player:
      infoset = infoset_generator(game_state, evt)

      # External sampling: choose a random action for the non-traversing player.
      pot_size = evt["round_state"]["pot"]["main"]["amount"]
      actions, mask = action_generator(evt["valid_actions"], pot_size)
      del evt, pot_size

      if t == 0:
        action_probs = strategies[other_player].get_action_probabilities_uniform()
      else:
        action_probs = strategies[other_player].get_action_probabilities(infoset, mask)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      # Add the action probabilities to the strategy buffer.
      if strt_mem is not None:
        strt_mem.add(infoset, action_probs, t)

      # EXTERNAL SAMPLING: choose only ONE action for the non-traversal player.
      if do_external_sampling:
        action, amount = actions[torch.multinomial(action_probs, 1).item()]
        updated_state, new_events = emulator.apply_action(game_state, action, amount)

        # NOTE(milo): Delete all events except the last one to save memory usage.
        return traverse(updated_state, new_events, emulator, action_generator, infoset_generator, traverse_player,
                        strategies, advt_mem, strt_mem, t, recursion_ctr=recursion_ctr,
                        do_external_sampling=do_external_sampling)
        
      # NO SAMPLING: Otherwise recurse on ALL non-traversal player actions.
      else:
        action_values = torch.zeros(2, len(actions))
        br_values = torch.zeros(2, len(actions))
        instant_regrets = torch.zeros(len(actions))

        plyr_idx = TreeNodeInfo.uuid_to_index(uuid)
        opp_idx = (1 - plyr_idx)

        for i, a in enumerate(actions):
          if mask[i] == 0:
            continue 
          updated_state, new_events = emulator.apply_action(game_state, a[0], a[1])
          child_node_info = traverse(updated_state, new_events, emulator, action_generator, infoset_generator,
                              traverse_player, strategies, advt_mem, strt_mem, t,
                              recursion_ctr=recursion_ctr, do_external_sampling=do_external_sampling)
          # Expected value of the acting player taking this action (for P1 and P2).
          action_values[:,i] = child_node_info.strategy_ev

          # Expected value for each player if the acting player takes this action and then they both
          # follow a best-response strategy.
          br_values[:,i] = child_node_info.best_response_ev
        
        # Sum along every action multiplied by its probability of occurring.
        node_info.strategy_ev = (action_values * action_probs).sum(axis=1)

        # Compute the instantaneous regrets for the traversing player.
        instant_regrets_tp = (action_values[tp_index] - (node_info.strategy_ev[tp_index] * mask))

        # The acting player chooses the BEST action with probability 1, while the opponent best
        # response EV depends on the reach probability of their next acting situation.
        node_info.best_response_ev[plyr_idx] = torch.max(br_values[plyr_idx,:])
        node_info.best_response_ev[opp_idx] = torch.sum(action_probs * br_values[opp_idx,:])

        # Exploitability is the difference in payoff between a local best response strategy and the
        # full mixed strategy.
        node_info.exploitability = node_info.best_response_ev - node_info.strategy_ev
        
        return node_info
    
    else:
      raise NotImplementedError()
