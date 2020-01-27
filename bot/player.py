import time

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from pokerbots_cpp import pokerbots_cpp as cpp


class Player(Bot):
  def __init__(self):
    '''
    Called when a new game starts. Called exactly once.
    '''
    self._pf = cpp.PermutationFilter(20000)
    self._compute_ev_samples = 1   # Number of permutations to sample for EV calculation.
    self._compute_ev_iters = 100    # Number of MC iters to use in pbots_calc.

    self._num_showdowns_seen = 0
    self._num_showdowns_converge = 120

    # Store the computed EV on each street.
    self._street_ev = {}

  def handle_new_round(self, game_state, round_state, active):
    '''
    Called when a new round starts. Called NUM_ROUNDS times.

    Arguments:
    game_state: the GameState object.
    round_state: the RoundState object.
    active: your player's index.

    Returns:
    Nothing.
    '''
    my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
    game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
    round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
    my_cards = round_state.hands[active]  # your cards
    big_blind = bool(active)  # True if you are the big blind

    self._street_ev = {}

  def handle_round_over(self, game_state, terminal_state, active):
    '''
    Called when a round ends. Called NUM_ROUNDS times.

    Arguments:
    game_state: the GameState object.
    terminal_state: the TerminalState object.
    active: your player's index.

    Returns:
    Nothing.
    '''
    my_delta = terminal_state.deltas[active]  # your bankroll change from this round
    previous_state = terminal_state.previous_state  # RoundState before payoffs
    street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
    my_cards = previous_state.hands[active]  # your cards
    opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
    board_cards = previous_state.deck[:street]

    winner_hole_cards = None
    loser_hole_cards = None
    if my_delta > 0:
      winner_hole_cards = my_cards
      loser_hole_cards = opp_cards
    elif my_delta < 0:
      winner_hole_cards = opp_cards
      loser_hole_cards = my_cards
    
    # Check if we saw the opponent's cards in the showdown.
    if winner_hole_cards is not None and loser_hole_cards is not None and len(loser_hole_cards) > 0:
      self._num_showdowns_seen += 1
      if self._pf.Nonzero() <= 0 or self._num_showdowns_seen > self._num_showdowns_converge:
        if self._pf.Nonzero() <= 0: print("WARNING: PermutationFilter particles all died")
        if self._num_showdowns_seen > self._num_showdowns_converge:
          print("PermutationFilter CONVERGED")
        return
      
      result = cpp.ShowdownResult("".join(winner_hole_cards), "".join(loser_hole_cards), "".join(board_cards))
      print("Updating with showdown result")
      self._pf.Update(result)

  def get_action(self, game_state, round_state, active):
    '''
    Where the magic happens - your code should implement this function.
    Called any time the engine needs an action from your bot.

    Arguments:
    game_state: the GameState object.
    round_state: the RoundState object.
    active: your player's index.

    Returns:
    Your action.
    '''
    legal_actions = round_state.legal_actions()  # the actions you are allowed to take

    street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
    my_cards = round_state.hands[active]  # your cards
    board_cards = round_state.deck[:street]  # the board cards
    my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
    opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
    my_stack = round_state.stacks[active]  # the number of chips you have remaining
    opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
    continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
    my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
    opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

    # If we haven't computed EV yet on this street, do so.
    if street not in self._street_ev:
      self._street_ev[street] = self._pf.ComputeEvRandom(
          "".join(my_cards), "".join(board_cards), "", self._compute_ev_samples, self._compute_ev_iters)

    EV = self._street_ev[street]

    if self._pf.Nonzero() <= 0:
      print("Particle filter empty, checkfolding")
      return CheckAction if CheckAction in legal_actions else FoldAction()

    pot_size = my_contribution + opp_contribution

    if RaiseAction in legal_actions:
      min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
      min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
      max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

    # CASE 1: Don't need to call.
    if CheckAction in legal_actions:
      if EV <= 0.5 or RaiseAction not in legal_actions:
        return CheckAction()
      elif EV <= 0.8:
        return RaiseAction(min(max(pot_size, min_raise), max_raise))
      else:
        return RaiseAction(max_raise)
    
    # CASE 2: Must call to continue.
    else:
      pot_after_call = 2 * opp_contribution
      equity = EV * pot_after_call

      # If calling costs more than the expected payout after calling, fold.
      if equity < continue_cost:
        return FoldAction()
      else:
        # Do a pot raise.
        if EV >= 0.8:
          return RaiseAction(min(max(pot_after_call, min_raise), max_raise)) if RaiseAction in legal_actions else CallAction()
        elif EV >= 0.9:
          return RaiseAction(min(max(2 * pot_after_call, min_raise), max_raise)) if RaiseAction in legal_actions else CallAction()
        else:
          return CallAction()

if __name__ == '__main__':
  run_bot(Player(), parse_args())
