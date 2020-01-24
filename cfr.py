import pickle, os

import torch
from constants import Constants
from engine import *
from utils import apply_mask_and_normalize, apply_mask_and_uniform
from traverse import create_new_round, make_actions, make_infoset, get_street_0123, make_precomputed_ev
from infoset import bucket_small, bucket_small_join


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


class RegretMatchedStrategy(object):
  def __init__(self):
    self._regrets = {}

  def size(self):
    return len(self._regrets)

  def add_regret(self, infoset, r):
    """
    Adds an instantaneous regret to total regret.
    """
    assert(len(r) == Constants.NUM_ACTIONS)

    bucket = bucket_small(infoset)
    bstring = bucket_small_join(bucket)

    if bstring not in self._regrets:
      self._regrets[bstring] = torch.zeros(Constants.NUM_ACTIONS)

    self._regrets[bstring] += r

  def get_strategy(self, infoset, valid_mask):
    """
    Does regret matching to return a probabilistic strategy.
    """
    bucket = bucket_small(infoset)
    bstring = bucket_small_join(bucket)

    if bstring not in self._regrets:
      self._regrets[bstring] = torch.zeros(Constants.NUM_ACTIONS)

    total_regret = self._regrets[bstring].clone()

    with torch.no_grad():
      r_plus = torch.clamp(total_regret, min=0)

      # As advocated by Brown et. al., choose the action with highest advantage when all of them are
      # less than zero.
      # if r_plus.sum() < 1e-5:
      #   total_regret -= total_regret.min()      # Make nonnegative.
      #   total_regret *= valid_mask              # Mask out illegal actions.
      #   r = torch.zeros(Constants.NUM_ACTIONS)  # Probability 1 for best action.  
      #   r[torch.argmax(total_regret)] = 1.0
      # else:
      #   r = r_plus
      # If no positive regrets, return a UNIFORM strategy.
      if r_plus.sum() < 1e-3:
        r = torch.ones(Constants.NUM_ACTIONS)
      else:
        r = r_plus

      return r / r.sum()

  def save(self, filename):
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    with open(filename, "wb") as f:
      pickle.dump(self._regrets, f)
    print("Saved RegretMatchedStrategy to {}".format(filename))

  def load(self, filename):
    with open(filename, "rb") as f:
      self._regrets = pickle.load(f)
    print("Loaded {} items from {}".format(self.size(), filename))

  def merge_and_save(self, filename, lock):
    lock.acquire()

    existing_regrets = {}
    if os.path.exists(filename):
      print("[MERGE] File already exists, loading and combining with myself")
      with open(filename, "rb") as f:
        existing_regrets = pickle.load(f)
    
    print("[MERGE] Merging {} existing with my {}".format(len(existing_regrets), self.size()))
    for key in self._regrets:
      if key not in existing_regrets:
        existing_regrets[key] = torch.zeros(Constants.NUM_ACTIONS)
      existing_regrets[key] += self._regrets[key]

    print("[MERGE] Total of {} regrets after merge".format(len(existing_regrets)))

    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    with open(filename, "wb") as f:
      pickle.dump(existing_regrets, f)

    print("[MERGE] Done with merge, releasing lock")
    lock.release()


# def traverse_cfr(round_state, traverse_plyr, sb_plyr_idx, regrets, strategies, t,
#                  reach_probabilities, precomputed_ev, rctr=[0], allow_updates=True,
#                  do_external_sampling=True):
#   """
#   Traverse the game tree with external and chance sampling.

#   NOTE: Only the traverse player updates their regrets. When the non-traverse player acts,
#   they add their strategy to the average strategy.
#   """
#   with torch.no_grad():
#     node_info = TreeNodeInfo()

#     rctr[0] += 1
#     other_plyr = (1 - traverse_plyr)
  
#     #================== TERMINAL NODE ====================
#     if isinstance(round_state, TerminalState):
#       node_info.strategy_ev = torch.Tensor(round_state.deltas)      # TODO: make sure this is correct.
#       # There are no choices to make here; the best response payoff is the outcome.
#       node_info.best_response_ev = node_info.strategy_ev
#       return node_info

#     active_plyr_idx = round_state.button % 2
#     is_traverse_player_action = (active_plyr_idx == traverse_plyr)

#     #============== TRAVERSE PLAYER ACTION ===============
#     if is_traverse_player_action:
#       infoset = make_infoset(round_state, traverse_plyr, (traverse_plyr == sb_plyr_idx), precomputed_ev)
#       actions, mask = make_actions(round_state)

#       # Do regret matching to get action probabilities.
#       action_probs = regrets[traverse_plyr].get_strategy(infoset, mask)
#       action_probs = apply_mask_and_uniform(action_probs, mask)
#       assert torch.allclose(action_probs.sum(), torch.ones(1), rtol=1e-3, atol=1e-3)

#       action_values = torch.zeros(2, len(actions))     # Expected payoff if we take an action and play according to sigma.
#       br_values = torch.zeros(2, len(actions))         # Expected payoff if we take an action and play according to BR.
#       immediate_regrets = torch.zeros(len(actions))    # Regret for not choosing an action over the current strategy.

#       for i, a in enumerate(actions):
#         if action_probs[i].item() <= 0: # NOTE: this should handle masked actions also.
#           continue
#         assert(mask[i] > 0)
#         next_round_state = round_state.copy().proceed(a)
#         next_reach_prob = reach_probabilities.clone()
#         next_reach_prob[traverse_plyr] *= action_probs[i]
#         child_node_info = traverse_cfr(
#             next_round_state, traverse_plyr, sb_plyr_idx, regrets,
#             strategies, t, next_reach_prob, precomputed_ev,
#             rctr=rctr, allow_updates=allow_updates, do_external_sampling=do_external_sampling)

#         action_values[:,i] = child_node_info.strategy_ev
#         br_values[:,i] = child_node_info.best_response_ev
      
#       # Sum along every action multiplied by its probability of occurring.
#       node_info.strategy_ev = (action_values * action_probs).sum(axis=1)
#       immediate_regrets_tp = mask * (action_values[traverse_plyr] - node_info.strategy_ev[traverse_plyr])

#       # Best response strategy: the acting player chooses the BEST action with probability 1.
#       node_info.best_response_ev[traverse_plyr] = torch.max(br_values[traverse_plyr,:])
#       node_info.best_response_ev[other_plyr] = torch.sum(action_probs * br_values[other_plyr,:])

#       # Exploitability is the difference in payoff between a local best response strategy and the full mixed strategy.
#       node_info.exploitability = (node_info.best_response_ev - node_info.strategy_ev)

#       if allow_updates:
#         # TODO: some conflicting info about which reach prob should multiply the avg strategy
#         strategies[traverse_plyr].add_regret(infoset, reach_probabilities[traverse_plyr] * action_probs)
#         regrets[traverse_plyr].add_regret(infoset, reach_probabilities[other_plyr] * immediate_regrets_tp)

#       return node_info

#     #================== NON-TRAVERSE PLAYER ACTION =================
#     else:
#       if do_external_sampling:
#         infoset = make_infoset(round_state, other_plyr, (other_plyr == sb_plyr_idx), precomputed_ev)

#         # External sampling: choose a random action for the non-traversing player.
#         actions, mask = make_actions(round_state)
#         action_probs = regrets[other_plyr].get_strategy(infoset, mask)
#         action_probs = apply_mask_and_uniform(action_probs, mask)
#         assert torch.allclose(action_probs.sum(), torch.ones(1), rtol=1e-3, atol=1e-3)

#         # EXTERNAL SAMPLING: choose only ONE action for the non-traversal player.
#         action = actions[torch.multinomial(action_probs, 1).item()]
#         next_round_state = round_state.copy().proceed(action)
#         next_reach_prob = reach_probabilities.clone()
#         return traverse_cfr(next_round_state, traverse_plyr, sb_plyr_idx, regrets,
#                             strategies, t, reach_probabilities, precomputed_ev,
#                             rctr=rctr, allow_updates=allow_updates,
#                             do_external_sampling=do_external_sampling)

#       else:
#         infoset = make_infoset(round_state, other_plyr, (other_plyr == sb_plyr_idx), precomputed_ev)
#         actions, mask = make_actions(round_state)

#         # Do regret matching to get action probabilities.
#         action_probs = regrets[other_plyr].get_strategy(infoset, mask)
#         action_probs = apply_mask_and_uniform(action_probs, mask)
#         assert torch.allclose(action_probs.sum(), torch.ones(1), rtol=1e-3, atol=1e-3)

#         action_values = torch.zeros(2, len(actions))     # Expected payoff if we take an action and play according to sigma.
#         br_values = torch.zeros(2, len(actions))         # Expected payoff if we take an action and play according to BR.
#         immediate_regrets = torch.zeros(len(actions))    # Regret for not choosing an action over the current strategy.

#         for i, a in enumerate(actions):
#           if action_probs[i].item() <= 0: # NOTE: this should handle masked actions also.
#             continue
#           assert(mask[i] > 0)
#           next_round_state = round_state.copy().proceed(a)
#           next_reach_prob = reach_probabilities.clone()
#           next_reach_prob[other_plyr] *= action_probs[i]
#           child_node_info = traverse_cfr(
#               next_round_state, traverse_plyr, sb_plyr_idx, regrets,
#               strategies, t, next_reach_prob, precomputed_ev,
#               rctr=rctr, allow_updates=allow_updates, do_external_sampling=do_external_sampling)

#           action_values[:,i] = child_node_info.strategy_ev
#           br_values[:,i] = child_node_info.best_response_ev
        
#         # Sum along every action multiplied by its probability of occurring.
#         node_info.strategy_ev = (action_values * action_probs).sum(axis=1)
#         immediate_regrets_tp = mask * (action_values[other_plyr] - node_info.strategy_ev[other_plyr])

#         # Best response strategy: the acting player chooses the BEST action with probability 1.
#         node_info.best_response_ev[other_plyr] = torch.max(br_values[other_plyr,:])
#         node_info.best_response_ev[traverse_plyr] = torch.sum(action_probs * br_values[traverse_plyr,:])

#         # Exploitability is the difference in payoff between a local best response strategy and the full mixed strategy.
#         node_info.exploitability = (node_info.best_response_ev - node_info.strategy_ev)

#         # TODO(milo): Should the non-traverse player be updated also???
#         if allow_updates:
#           # TODO: some conflicting info about which reach prob should multiply the avg strategy
#           strategies[other_plyr].add_regret(infoset, reach_probabilities[other_plyr] * action_probs)
#           regrets[other_plyr].add_regret(infoset, reach_probabilities[traverse_plyr] * immediate_regrets_tp)

#         return node_info


def traverse_cfr(round_state, traverse_plyr, sb_plyr_idx, regrets, strategies, t,
                 reach_probabilities, precomputed_ev, rctr=[0], allow_updates=True,
                 do_external_sampling=True):
  """
  Traverse the game tree with external and chance sampling.

  NOTE: Only the traverse player updates their regrets. When the non-traverse player acts,
  they add their strategy to the average strategy.
  """
  with torch.no_grad():
    node_info = TreeNodeInfo()
    rctr[0] += 1
  
    #================== TERMINAL NODE ====================
    if isinstance(round_state, TerminalState):
      node_info.strategy_ev = torch.Tensor(round_state.deltas) # There are no choices to make here; the best response payoff is the outcome.
      node_info.best_response_ev = node_info.strategy_ev
      return node_info

    # print("Button:", round_state.button)
    active_plyr_idx = round_state.button % 2
    inactive_plyr_idx = (1 - active_plyr_idx)

    infoset = make_infoset(round_state, active_plyr_idx, (active_plyr_idx == sb_plyr_idx), precomputed_ev)
    actions, mask = make_actions(round_state)
    # print(active_plyr_idx, bucket_small_join(bucket_small(infoset)), mask)

    # Do regret matching to get action probabilities.
    action_probs = regrets[active_plyr_idx].get_strategy(infoset, mask)
    action_probs = apply_mask_and_uniform(action_probs, mask)
    assert torch.allclose(action_probs.sum(), torch.ones(1), rtol=1e-3, atol=1e-3)
    # print(bucket_small_join(bucket_small(infoset)))
    # print("Acting: {} probs={} mask={}".format(active_plyr_idx, action_probs, mask))

    action_values = torch.zeros(2, len(actions))     # Expected payoff if we take an action and play according to sigma.
    br_values = torch.zeros(2, len(actions))         # Expected payoff if we take an action and play according to BR.
    immediate_regrets = torch.zeros(len(actions))    # Regret for not choosing an action over the current strategy.

    if active_plyr_idx != traverse_plyr and do_external_sampling:
      assert(False)
      # EXTERNAL SAMPLING: choose only ONE action for the non-traversal player.
      action = actions[torch.multinomial(action_probs, 1).item()]
      next_round_state = round_state.copy().proceed(action)
      next_reach_prob = reach_probabilities.clone()
      return traverse_cfr(next_round_state, traverse_plyr, sb_plyr_idx, regrets,
                          strategies, t, reach_probabilities, precomputed_ev,
                          rctr=rctr, allow_updates=allow_updates,
                          do_external_sampling=do_external_sampling)
    
    else:
      for i, a in enumerate(actions):
        if action_probs[i].item() <= 0: # NOTE: this should handle masked actions also.
          continue
        assert(mask[i] > 0)
        next_round_state = round_state.copy().proceed(a)
        next_reach_prob = reach_probabilities.clone()
        next_reach_prob[active_plyr_idx] *= action_probs[i]
        child_node_info = traverse_cfr(
            next_round_state, traverse_plyr, sb_plyr_idx, regrets,
            strategies, t, next_reach_prob, precomputed_ev,
            rctr=rctr, allow_updates=allow_updates, do_external_sampling=do_external_sampling)

        action_values[:,i] = child_node_info.strategy_ev
        br_values[:,i] = child_node_info.best_response_ev
      
      # Sum along every action multiplied by its probability of occurring.
      node_info.strategy_ev = (action_values * action_probs).sum(axis=1)
      immediate_regrets_tp = mask * (action_values[active_plyr_idx] - node_info.strategy_ev[active_plyr_idx])

      # Best response strategy: the acting player chooses the BEST action with probability 1.
      node_info.best_response_ev[active_plyr_idx] = torch.max(br_values[active_plyr_idx,:])
      node_info.best_response_ev[inactive_plyr_idx] = torch.sum(action_probs * br_values[inactive_plyr_idx,:])

      # Exploitability is the difference in payoff between a local best response strategy and the full mixed strategy.
      node_info.exploitability = (node_info.best_response_ev - node_info.strategy_ev)

      if allow_updates:
        # TODO: what should reach prob be?
        strategies[active_plyr_idx].add_regret(infoset, reach_probabilities[active_plyr_idx] * action_probs)
        regrets[active_plyr_idx].add_regret(infoset, reach_probabilities[inactive_plyr_idx] * immediate_regrets_tp)
        # strategies[active_plyr_idx].add_regret(infoset, action_probs)
        # regrets[active_plyr_idx].add_regret(infoset, immediate_regrets_tp)

      return node_info
