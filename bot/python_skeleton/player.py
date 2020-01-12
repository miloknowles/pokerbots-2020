import time

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from permutation_filter import *


class Player(Bot):
    '''
    A pokerbot.
    '''
    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self._pf = PermutationFilter(10000)

        self._nparticles = 500
        self._resample_thresh = 500
        # self._next_resample_nparticles = 500

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

        print("[LOG] [NR] bankroll={} game_clock={} round_num={} my_cards={} big_blind={}".format(
            my_bankroll,
            game_clock,
            round_num,
            my_cards,
            big_blind
        ))

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

        print("[LOG] [RO] my_delta={} street={} my_cards={} opp_cards={}".format(
            my_delta,
            street,
            my_cards,
            opp_cards
        ))

        winner_hole_cards = None
        loser_hole_cards = None

        if my_delta > 0:
            winner_hole_cards = my_cards
            loser_hole_cards = opp_cards
        elif my_delta < 0:
            winner_hole_cards = opp_cards
            loser_hole_cards = my_cards
        
        if winner_hole_cards is not None and loser_hole_cards is not None and len(loser_hole_cards) > 0:
            result = ShowdownResult(winner_hole_cards, loser_hole_cards, board_cards)
            print(result)
            t0 = time.time()
            self._pf.update(result)
            elapsed = time.time() - t0
            print("Updated filter in {} sec, UNIQUE={}".format(elapsed, self._pf.unique()))

            if self._pf.nonzero() < self._resample_thresh:
                t0 = time.time()
                self._pf.resample(self._nparticles)
                elapsed = time.time() - t0
                unique = self._pf.unique()
                print("Did resample({}) in {} sec, UNIQUE={}".format(self._nparticles, elapsed, unique))
                # self._next_resample_nparticles = max(500, min(int(1.5 * unique), 1000))

                if unique <= 5:
                    print("\n ================= FILTER CONVERGED ================")
                    for p in self._pf.get_unique_permutations():
                        print(p)

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
        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
