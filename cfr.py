import pickle, os

import torch
from constants import Constants
from engine import *
from utils import apply_mask_and_normalize
from traverse import create_new_round, make_actions, make_infoset, get_street_0123, make_precomputed_ev, TreeNodeInfo
from infoset import bucket_small, bucket_small_join


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
      if r_plus.sum() < 1e-5:
        total_regret -= total_regret.min()       # Make nonnegative.
        total_regret *= valid_mask              # Mask out illegal actions.
        r = torch.zeros(Constants.NUM_ACTIONS)  # Probability 1 for best action.  
        r[torch.argmax(total_regret)] = 1.0
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


def traverse_cfr(round_state, traverse_plyr_idx, sb_plyr_idx, regrets, avg_strategy, t, precomputed_ev, rctr=[0]):
  """
  Traverse the game tree with external and chance sampling.

  NOTE: Only the traverse player updates their regrets. When the non-traverse player acts,
  they add their strategy to the average strategy.
  """
  with torch.no_grad():
    node_info = TreeNodeInfo()

    rctr[0] += 1
    other_player_idx = (1 - traverse_plyr_idx)
  
    #================== TERMINAL NODE ====================
    if isinstance(round_state, TerminalState):
      node_info.strategy_ev = torch.Tensor(round_state.deltas)
      node_info.best_response_ev = node_info.strategy_ev
      return node_info

    active_plyr_idx = round_state.button % 2
    is_traverse_player_action = (active_plyr_idx == traverse_plyr_idx)

    #============== TRAVERSE PLAYER ACTION ===============
    if is_traverse_player_action:
      infoset = make_infoset(round_state, traverse_plyr_idx, (traverse_plyr_idx == sb_plyr_idx), precomputed_ev)
      actions, mask = make_actions(round_state)

      # Do regret matching to get action probabilities.
      action_probs = regrets[traverse_plyr_idx].get_strategy(infoset, mask)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      action_values = torch.zeros(2, len(actions))
      br_values = torch.zeros(2, len(actions))
      instant_regrets = torch.zeros(len(actions))

      plyr_idx = traverse_plyr_idx
      opp_idx = (1 - plyr_idx)

      for i, a in enumerate(actions):
        if mask[i] <= 0:
          continue
        next_round_state = round_state.copy().proceed(a)
        child_node_info = traverse_cfr(next_round_state, traverse_plyr_idx, sb_plyr_idx, regrets, avg_strategy, t, precomputed_ev, rctr=rctr)
        
        # Expected value of the acting player taking this action and then continuing according to their strategy.
        action_values[:,i] = child_node_info.strategy_ev

        # Expected value for each player if the acting player takes this action and then they both
        # follow a best-response strategy.
        br_values[:,i] = child_node_info.best_response_ev
      
      # Sum along every action multiplied by its probability of occurring.
      node_info.strategy_ev = (action_values * action_probs).sum(axis=1)

      # Compute the instantaneous regrets for the traversing player.
      instant_regrets_tp = mask * (action_values[traverse_plyr_idx] - node_info.strategy_ev[traverse_plyr_idx])

      # The acting player chooses the BEST action with probability 1, while the opponent best
      # response EV depends on the reach probability of their next acting situation.
      node_info.best_response_ev[plyr_idx] = torch.max(br_values[plyr_idx,:])
      node_info.best_response_ev[opp_idx] = torch.sum(action_probs * br_values[opp_idx,:])

      # Exploitability is the difference in payoff between a local best response strategy and the
      # full mixed strategy.
      node_info.exploitability = node_info.best_response_ev - node_info.strategy_ev

      # Add the instantaneous regrets to advantage memory for the traversing player.
      regrets[traverse_plyr_idx].add_regret(infoset, instant_regrets_tp)

      return node_info

    #================== NON-TRAVERSE PLAYER ACTION =================
    else:
      infoset = make_infoset(round_state, other_player_idx, (other_player_idx == sb_plyr_idx), precomputed_ev)

      # External sampling: choose a random action for the non-traversing player.
      actions, mask = make_actions(round_state)
      action_probs = regrets[other_player_idx].get_strategy(infoset, mask)
      action_probs = apply_mask_and_normalize(action_probs, mask)
      assert torch.allclose(action_probs.sum(), torch.ones(1))

      # Add the action probabilities to the average strategy buffer.
      avg_strategy.add_regret(infoset, action_probs)

      # EXTERNAL SAMPLING: choose only ONE action for the non-traversal player.
      action = actions[torch.multinomial(action_probs, 1).item()]
      next_round_state = round_state.copy().proceed(action)

      return traverse_cfr(next_round_state, traverse_plyr_idx, sb_plyr_idx, regrets, avg_strategy, t, precomputed_ev, rctr=rctr)

