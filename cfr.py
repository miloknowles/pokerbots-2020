import pickle, os

import torch
import eval7

from constants import Constants
from engine import *
from utils import apply_mask_and_normalize, apply_mask_and_uniform
from infoset import bucket_small, bucket_small_join, EvInfoSet
from pbots_calc import calc, CalcWithLookup


EV_CALCULATOR = CalcWithLookup()


def get_street_0123(s):
  return 0 if s == 0 else s - 2


def make_precomputed_ev(round_state):
  out = {}
  for s in (0, 3, 4, 5):
    if s == 3:
      iters = 5000
    elif s == 4:
      iters = 5000
    elif s == 5:
      iters = 1326
    else:
      iters = 1

    h1 = [str(round_state.hands[0][0]), str(round_state.hands[0][1])]
    h2 = [str(round_state.hands[1][0]), str(round_state.hands[1][1])]
    board = str.encode("".join([str(c) for c in round_state.deck.peek(s)]))

    ev1 = EV_CALCULATOR.calc(h1, board, b"", iters)
    ev2 = EV_CALCULATOR.calc(h2, board, b"", iters)
    out[s] = [ev1, ev2]

  return out


def make_infoset(round_state, player_idx, player_is_sb, precomputed_ev=None):
  """
  Make an information set representation of the game state.

  round_state (RoundState) : From MIT game engine.
  player_idx (int) : 0 if P1 is acting, 1 if P2 is acting.
  player_is_sb (bool) : Is the acting player the SB?
  """
  h = torch.zeros(2 + Constants.BET_HISTORY_SIZE)
  for street, actions in enumerate(round_state.bet_history):
    offset = street * Constants.BET_ACTIONS_PER_STREET + (2 if street > 0 else 0)
    for i, add_amt in enumerate(actions):
      if street > 0:
        i = min(i, Constants.BET_ACTIONS_PER_STREET - 2 + i % 2)
      else:
        bet_actions_preflop = Constants.BET_ACTIONS_PER_STREET + 2
        i = min(i, bet_actions_preflop - 2 + i % 2)
      h[offset + i] += add_amt

  if precomputed_ev is not None:
    ev = precomputed_ev[round_state.street][player_idx]
    return EvInfoSet(ev, h, 0 if player_is_sb else 1, get_street_0123(round_state.street))
  
  else:
    # hand = "{}:xx".format(str(round_state.hands[player_idx][0]) + str(round_state.hands[player_idx][1]))
    hand = [str(round_state.hands[player_idx][0]), str(round_state.hands[player_idx][1])]
    board = "".join([str(c) for c in round_state.deck.peek(round_state.street)])

    # Use fewer iters on later streets.
    if len(board) == 6:
      iters = 5000
    elif len(board) == 8:
      iters = 4000
    elif len(board) == 10:
      iters = 1326
    else:
      iters = 1

    ev = EV_CALCULATOR.calc(hand, str.encode(board), b"", iters)
    # ev = calc(str.encode(hand), str.encode(board), b"", 1000).ev[0]

    return EvInfoSet(ev, h, 0 if player_is_sb else 1, get_street_0123(round_state.street))


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

  # If we've exceeded the number of betting actions, or if doing this action would hit the limit,
  # disable raising, forcing the bot to either fold or call.
  bet_actions_this_street = len(round_state.bet_history[get_street_0123(round_state.street)])
  force_fold_call = bet_actions_this_street >= (Constants.BET_ACTIONS_PER_STREET - 1)

  actions_mask = torch.zeros(len(Constants.ALL_ACTIONS))
  actions_unscaled = deepcopy(Constants.ALL_ACTIONS)

  for i, a in enumerate(actions_unscaled):
    if type(a) in valid_action_set and not (isinstance(a, RaiseAction) and force_fold_call):
      actions_mask[i] = 1
    if isinstance(a, RaiseAction):
      pot_size_after_call = pot_size + abs(round_state.pips[0] - round_state.pips[1])
      amt_to_add = a.amount * pot_size_after_call
      amt_to_raise = max(round_state.pips[0], round_state.pips[1]) + amt_to_add
      amt = min(max_raise, max(min_raise, amt_to_raise))
      actions_unscaled[i] = RaiseAction(amt)

  assert(len(actions_unscaled) == len(actions_mask))
  return actions_unscaled, actions_mask


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
  round_state = RoundState(button_player, 0, pips, stacks, hands, deck, None, [[1, 2]], button_player)
  return round_state


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

    active_plyr_idx = round_state.button % 2
    inactive_plyr_idx = (1 - active_plyr_idx)

    infoset = make_infoset(round_state, active_plyr_idx, (active_plyr_idx == sb_plyr_idx), precomputed_ev)
    actions, mask = make_actions(round_state)

    # Do regret matching to get action probabilities.
    action_probs = regrets[active_plyr_idx].get_strategy(infoset, mask)
    action_probs = apply_mask_and_uniform(action_probs, mask)
    assert torch.allclose(action_probs.sum(), torch.ones(1), rtol=1e-3, atol=1e-3)

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
        # if action_probs[i].item() <= 0: # NOTE: this should handle masked actions also.
          # continue
        if mask[i].item() <= 0:
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

      if allow_updates and active_plyr_idx == traverse_plyr:
        # NOTE: Zinkevich et. al. multiple the immediate regret by the opponent reach probability,
        # and the strategy by the player reach probability.
        strategies[active_plyr_idx].add_regret(infoset, reach_probabilities[active_plyr_idx] * action_probs)
        regrets[active_plyr_idx].add_regret(infoset, reach_probabilities[inactive_plyr_idx] * immediate_regrets_tp)

      return node_info
