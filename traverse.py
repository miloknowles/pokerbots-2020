import torch

import numpy as np
from copy import deepcopy
import time

from utils import *
from constants import Constants
from utils import sample_uniform_action
from infoset import EvInfoSet


def traverse(round_state, action_generator, infoset_generator, traverse_player_idx, sb_player_idx,
             strategies, advt_mem, strt_mem, t, precomputed_ev, recursion_ctr=[0]):
  with torch.no_grad():
    node_info = TreeNodeInfo()

    recursion_ctr[0] += 1
    other_player_idx = (1 - traverse_player_idx)
  
    #================== TERMINAL NODE ====================
    if isinstance(round_state, TerminalState):
      node_info.strategy_ev = torch.Tensor(round_state.deltas)
      node_info.best_response_ev = node_info.strategy_ev
      # print("TERMINAL STATE")
      # print("*** strategy_ev=", node_info.strategy_ev)
      # print("*** best_response_ev=", node_info.best_response_ev)
      return node_info

    active_player_idx = round_state.button % 2
    is_traverse_player_action = (active_player_idx == traverse_player_idx)

    #============== TRAVERSE PLAYER ACTION ===============
    if is_traverse_player_action:
      infoset = infoset_generator(round_state, traverse_player_idx, traverse_player_idx == sb_player_idx, precomputed_ev)
      actions, mask = action_generator(round_state)

      # Do regret matching to get action probabilities.
      if t == 0:
        action_probs = strategies[traverse_player_idx].get_action_probabilities_uniform()
      else:
        action_probs = strategies[traverse_player_idx].get_action_probabilities(infoset, mask)

      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      action_values = torch.zeros(2, len(actions))
      br_values = torch.zeros(2, len(actions))
      instant_regrets = torch.zeros(len(actions))

      plyr_idx = traverse_player_idx
      opp_idx = (1 - plyr_idx)

      for i, a in enumerate(actions):
        if mask[i] <= 0:
          continue
        next_round_state = round_state.copy().proceed(a)
        # print("TRAVERSE ACTION:", a)
        child_node_info = traverse(next_round_state,
                                   action_generator, infoset_generator,
                                   traverse_player_idx, sb_player_idx, strategies, advt_mem, strt_mem, t,
                                   precomputed_ev, recursion_ctr=recursion_ctr)
        
        # Expected value of the acting player taking this action and then continuing according to their strategy.
        action_values[:,i] = child_node_info.strategy_ev

        # Expected value for each player if the acting player takes this action and then they both
        # follow a best-response strategy.
        br_values[:,i] = child_node_info.best_response_ev
      
      # Sum along every action multiplied by its probability of occurring.
      node_info.strategy_ev = (action_values * action_probs).sum(axis=1)
      
      # print("====> TRAVERSE PLAYER ACTION NODE <=====")
      # print(round_state.bet_history)
      # print("strategy_ev=", node_info.strategy_ev)

      # Compute the instantaneous regrets for the traversing player.
      instant_regrets_tp = mask * (action_values[traverse_player_idx] - node_info.strategy_ev[traverse_player_idx])

      # print("action_values=", action_values[traverse_player_idx])
      # print("instant_regrets=", instant_regrets_tp)

      # The acting player chooses the BEST action with probability 1, while the opponent best
      # response EV depends on the reach probability of their next acting situation.
      node_info.best_response_ev[plyr_idx] = torch.max(br_values[plyr_idx,:])
      node_info.best_response_ev[opp_idx] = torch.sum(action_probs * br_values[opp_idx,:])

      # print("best_response_plyr=", node_info.best_response_ev[plyr_idx])
      # print("best_response_opp=", node_info.best_response_ev[opp_idx])

      # Exploitability is the difference in payoff between a local best response strategy and the
      # full mixed strategy.
      node_info.exploitability = node_info.best_response_ev - node_info.strategy_ev
      # print("exploitability=", node_info.exploitability)

      # Add the instantaneous regrets to advantage memory for the traversing player.
      if advt_mem is not None:
        advt_mem.add(infoset, instant_regrets_tp, t)

      return node_info

    #================== NON-TRAVERSE PLAYER ACTION =================
    else:
      infoset = infoset_generator(round_state, other_player_idx, other_player_idx == sb_player_idx, precomputed_ev)

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
      next_round_state = round_state.copy().proceed(action)

      # print("NON-TRAVERSE ACTION:", action)

      return traverse(next_round_state,
                      action_generator, infoset_generator, traverse_player_idx, sb_player_idx,
                      strategies, advt_mem, strt_mem, t, precomputed_ev, recursion_ctr=recursion_ctr)
