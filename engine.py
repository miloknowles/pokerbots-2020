from numpy.random import geometric
from collections import namedtuple
from threading import Thread
from queue import Queue
from copy import deepcopy
import time
import json
import subprocess
import socket
import eval7
import sys
import os


SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 200


FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

STREET_NAMES = ['Flop', 'Turn', 'River']
DECODE = {'F': FoldAction, 'C': CallAction, 'K': CheckAction, 'R': RaiseAction}
CCARDS = lambda cards: ','.join(map(str, cards))
PCARDS = lambda cards: '{} [{}]'.format(' '.join(map(str, cards)), ' '.join(map(str, map(PERM.get, cards))))
PVALUE = lambda name, value: ', {} ({})'.format(name, value)
STATUS = lambda players: ''.join([PVALUE(p.name, p.bankroll) for p in players])


class RoundState(namedtuple('_RoundState', ['button', 'street', 'pips', 'stacks', 'hands', 'deck', 'previous_state', 'bet_history', 'sb_player'])):
    '''
    Encodes the game tree for one round of poker.
    '''
    def showdown(self):
        '''
        Compares the players' hands and computes payoffs.
        '''
        # score0 = eval7.evaluate(list(map(PERM.get, self.deck.peek(5) + self.hands[0])))
        # score1 = eval7.evaluate(list(map(PERM.get, self.deck.peek(5) + self.hands[1])))
        score0 = eval7.evaluate(self.deck.peek(5) + self.hands[0])
        score1 = eval7.evaluate(self.deck.peek(5) + self.hands[1])
        if score0 > score1:
            delta = STARTING_STACK - self.stacks[1]
        elif score0 < score1:
            delta = self.stacks[0] - STARTING_STACK
        else:  # split the pot
            delta = (self.stacks[0] - self.stacks[1]) // 2
        return TerminalState([delta, -delta], self)

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.
        '''
        if self.street == 5:
            return self.showdown()
        
        # Add a new street of actions to the bet history.
        self.bet_history.append([])

        new_street = 3 if self.street == 0 else self.street + 1

        # NOTE: The MIT engine starts all new streets with button = 1 (2nd player always).
        return RoundState(1 - self.sb_player, new_street, [0, 0], self.stacks, self.hands, self.deck, self, self.bet_history, self.sb_player)

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.
        '''
        active = self.button % 2
        if isinstance(action, FoldAction):
            delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
            return TerminalState([delta, -delta], self)
        if isinstance(action, CallAction):
            if self.button == 0:  # sb calls bb
                self.bet_history[-1].append(1)
                return RoundState(1, 0, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.deck, self, self.bet_history, self.sb_player)
            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self, self.bet_history, self.sb_player)

            # Update the betting history.
            self.bet_history[-1].append(contribution)

            return state.proceed_street()

        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0) or self.button > 1:  # both players acted
                self.bet_history[-1].append(0)
                return self.proceed_street()
        
            # let opponent act
            self.bet_history[-1].append(0)
            return RoundState(self.button + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self, self.bet_history, self.sb_player)
        # isinstance(action, RaiseAction)
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        
        # Update the betting history.
        self.bet_history[-1].append(contribution)

        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self, self.bet_history, self.sb_player)

    def copy(self):
        return RoundState(self.button, self.street, deepcopy(self.pips), deepcopy(self.stacks),
                          self.hands, self.deck, self.previous_state, deepcopy(self.bet_history), self.sb_player)
