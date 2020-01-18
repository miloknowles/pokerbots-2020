import ray
import torch

import numpy as np
from copy import deepcopy
import time

from utils import *
from constants import Constants
from utils import sample_uniform_action
from infoset import EvInfoSet

import eval7
from engine import RoundState, FoldAction, CallAction, CheckAction, RaiseAction, TerminalState


def make_infoset(round_state, player_is_sb):
  h =  torch.zeros(Constants.NUM_BETTING_ACTIONS)
  for street, actions in enumerate(round_state.bet_history):
    offset = street * Constants.NUM_BETTING_ACTIONS
    for i, add_amt in enumerate(actions):
      h[offset + i] = add_amt
  
  

  return EvInfoSet(ev, h, player_is_sb)


def make_actions(round_state):
  """
  Makes the actions that our network can take (Fold, Call, Check, PotRaise, TwoPotRaise, ThreePotRaise).
  NOTE: A pot BET means adding chips to the pot equal to the current pot size.
  NOTE: A pot RAISE means calling and THEN adding chips equal to the called pot size.
  NOTE: If the current pot is x, then a pot raise puts the pip at 3x, two pot raise puts the pip at 5x, three pot at 7x.
  """
  valid_action_set = round_state.legal_actions()
  min_raise, max_raise = round_state.raise_bounds()
  pot_size = 2*Constants.INITIAL_STACK - (round_state.stacks[0] + round_state.stacks[1])

  actions_mask = torch.zeros(len(Constants.ALL_ACTIONS))
  actions_unscaled = deepcopy(Constants.ALL_ACTIONS)

  for i,  a in enumerate(actions_unscaled):
    if type(a) in valid_action_set:
      actions_mask[i] = 1
    if isinstance(a, RaiseAction):
      pot_size_after_call = pot_size + abs(round_state.pips[0] - round_state.pips[1])
      amt_to_add = a.amount * pot_size_after_call
      amt_to_raise = max(round_state.pips[0], round_state.pips[1]) + amt_to_add
      amt = min(max_raise, max(min_raise, amt_to_raise))
      actions_unscaled[i] = RaiseAction(amt)

  assert(len(actions_unscaled) == len(actions_mask))
  return actions_unscaled, actions_mask


class TreeNodeInfo(object):
  def __init__(self):
    """
    NOTE: The zero index always refers to PLAYER1 and the 1th index is PLAYER2.
    """
    # Expected value for each player at this node if they play according to their current strategy.
    self.strategy_ev = torch.zeros(2)

    # Expected value for each player if they choose a best-response strategy given the other.
    self.best_response_ev = torch.zeros(2)

    # The difference in EV between the best response strategy and the current strategy.
    self.exploitability = torch.zeros(2)


def create_new_round(button_player):
  """
  Randomly generate a round_state to start a new round.
  button_player (int) : 0 if PLAYER1 should be button (small blind), 1 if PLAYER2.
  NOTE: Button should alternate every time.
  """
  deck = eval7.Deck()
  deck.shuffle()
  hands = [deck.deal(2), deck.deal(2)]
  sb = Constants.SMALL_BLIND_AMOUNT
  bb = 2*Constants.SMALL_BLIND_AMOUNT
  pips = [sb, bb]
  stacks = [Constants.INITIAL_STACK - sb, Constants.INITIAL_STACK - bb]
  if button_player == 1:
    pips.reverse()
    stacks.reverse()
  round_state = RoundState(button_player, 0, pips, stacks, hands, deck, None, [[1, 2]])
  return round_state


def traverse(round_state, action_generator, infoset_generator, traverse_player_idx, strategies,
             advt_mem, strt_mem, t, recursion_ctr=[0]):
  with torch.no_grad():
    node_info = TreeNodeInfo()

    recursion_ctr[0] += 1
    other_player_idx = (1 - traverse_player_idx)
  
    #================== TERMINAL NODE ====================
    if isinstance(round_state, TerminalState):
      node_info.strategy_ev = round_state.deltas.copy()
      node_info.best_response_ev = node_info.strategy_ev.copy()
      return node_info

    # TODO: confirm that this matches indices
    active_player_idx = round_state.button % 2
    is_traverse_player_action = (active_player_idx == traverse_player_idx)

    #============== TRAVERSE PLAYER ACTION ===============
    if is_traverse_player_action:
      # TODO
      infoset = infoset_generator(round_state)
      actions, mask = action_generator(round_state)

      # Do regret matching to get action probabilities.
      # if t == 0:
      action_probs = strategies[traverse_player].get_action_probabilities_uniform()
      # else:
        # action_probs = strategies[traverse_player].get_action_probabilities(infoset, mask)
   
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
        next_round_state = round_state.proceed(a)
        child_node_info = traverse(next_round_state, action_generator, infoset_generator,
                                   traverse_player_idx, strategies, advt_mem, strt_mem, t,
                                   recursion_ctr=recursion_ctr)
        
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

    #================== NON-TRAVERSE PLAYER ACTION =================
    else:
      infoset = infoset_generator(round_state)

      # External sampling: choose a random action for the non-traversing player.
      actions, mask = action_generator(round_state)
      if t == 0:
        action_probs = strategies[other_player_idx].get_action_probabilities_uniform()
      else:
        action_probs = strategies[other_player_idx].get_action_probabilities(infoset, mask)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      # Add the action probabilities to the strategy buffer.
      if strt_mem is not None:
        strt_mem.add(infoset, action_probs, t)

      # EXTERNAL SAMPLING: choose only ONE action for the non-traversal player.
      action = actions[torch.multinomial(action_probs, 1).item()]
      next_round_state = round_state.proceed(action)

      # NOTE(milo): Delete all events except the last one to save memory usage.
      return traverse(next_round_state, action_generator, infoset_generator, traverse_player_idx,
                      strategies, advt_mem, strt_mem, t, recursion_ctr=recursion_ctr)


if __name__ == "__main__":
  button_player = 0
  round_state = create_new_round(button_player)
  print("Initial state:", round_state)
