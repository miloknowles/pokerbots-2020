import numpy as np
import eval7
from collections import OrderedDict


RANK_STR_TO_VAL = OrderedDict({
  "2": 0,
  "3": 1,
  "4": 2,
  "5": 3,
  "6": 4,
  "7": 5,
  "8": 6,
  "9": 7,
  "T": 8,
  "J": 9,
  "Q": 10,
  "K": 11,
  "A": 12
})

RANK_VAL_TO_STR = OrderedDict({
  0: "2",
  1: "3",
  2: "4",
  3: "5",
  4: "6",
  5: "7",
  6: "8",
  7: "9",
  8: "T",
  9: "J",
  10: "Q",
  11: "K",
  12: "A"
})


RANKS_ORDERED = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


def str_to_rank_and_suit(card):
  return RANK_STR_TO_VAL[card[0]], card[1]


def rank_and_suit_to_str(rank, suit):
  return RANK_VAL_TO_STR[rank] + suit


class Permutation(object):
  def __init__(self, true_to_perm):
    """
    Defines a two-way mapping between true values and permutation values (0-13). The zeroth index
    represents a 2, and the 13th index represents an A.

    2 3 4 5 6 7 8 9 T J Q  K  A
    0 1 2 3 4 5 6 7 8 9 10 11 12

    true_to_perm (list of int) : Should have length 13.
    """
    self.true_to_perm = np.array(true_to_perm)
    self.perm_to_true = np.zeros(13)

    # Build the reverse mapping (from permutation value to true value).
    for i, mapped_val in enumerate(self.true_to_perm):
      self.perm_to_true[mapped_val] = i

  def map_str(self, cards_str):
    out = []
    for c in cards_str:
      rank, suit = str_to_rank_and_suit(c)
      out.append(rank_and_suit_to_str(self.perm_to_true[rank], suit))
    return out

  def __str__(self):
    cards = ["{}x".format(v) for v in RANKS_ORDERED]
    perm = self.map_str(cards)
    return ' ' + ' '.join(cards) + ' ' + '\n' + '[' + ' '.join(perm) + ']'


class Constraint(object):
  def __init__(self, predicate, a, b):
    """
    Defines a constraint on possible value permutations.
    NOTE: a and b values should be zero-start (range from 0-13).

    For example, Constraint(">", 5, 9) means that the true value of a 7 is greater than that of a J.
    """
    self.predicate = predicate
    self.a = a
    self.b = b

  def satisfied(perm):
    if self.predicate == ">" or "!<":
      return perm.perm_to_true[self.a] > perm.perm_to_true[self.b]
    elif self.predicate == "<" or "!>":
      return perm.perm_to_true[self.a] < perm.perm_to_true[self.b]
    elif self.predicate == "c" or self.predicate == "!c":
      is_after = perm.perm_to_true[self.a] == ((perm.perm_to_true[self.b] + 1) % 13)
      is_before = perm.perm_to_true[self.a] == ((perm.perm_to_true[self.b] - 1) % 13)
      return (not is_after and not is_before) if self.predicate == "!c" else (is_after or is_before)
    else:
      raise NotImplementedError()


class ShowdownResult(object):
  def __init__(self, winner_hole_cards, loser_hole_cards, board_cards):
    """
    All arguments are lists of card strings.
    """
    self.winner_hole_cards = winner_hole_cards
    self.loser_hole_cards = loser_hole_cards
    self.board_cards = board_cards

  def __str__(self):
    return "Winning hand: " + " ".join(self.winner_hole_cards) + " | " \
           "Losing hand: " + " ".join(self.loser_hole_cards) + " | " \
           "Board cards: " + " ".join(self.board_cards)


class PermutationFilter(object):
  def __init__(self, num_particles):
    self._num_particles = num_particles

    self._particles = [self.sample_no_constraints() for _ in range(self._num_particles)]
    self._weights = np.ones(self._num_particles) / self._num_particles

    # self._constraints = []
    self._results = []

  def update(self, result):
    for i, p in enumerate(self._particles):
      if not self.is_consistent_with_result(p, result):
        self._weights[i] = 0

    self._results.append(result)

  def is_consistent_with_result(self, p, result):
    board_and_winner_true = [eval7.Card(c) for c in p.map_str(result.board_cards + result.winner_hole_cards)]
    board_and_loser_true = [eval7.Card(c) for c in p.map_str(result.board_cards + result.loser_hole_cards)]

    score_winner = eval7.evaluate(board_and_winner_true)
    score_loser = eval7.evaluate(board_and_loser_true)

    # If the losing hand has a higher score than the winning score, then the permutation is
    # inconsistent with observations.
    if score_winner <= score_loser:
      return False
    else:
      return True

  def resample(self):
    """
    This takes about 30 sec for 1000 resamples w/ 1000 particles right now.
    """
    self._particles = []
    self._weights = np.zeros(self._num_particles)

    while len(self._particles) < self._num_particles:
      p = self.sample_no_constraints()

      is_valid = True
      for r in self._results:
        if not self.is_consistent_with_result(p, r):
          is_valid = False
          break
      
      if is_valid:
        self._particles.append(p)

  def nonzero(self):
    """
    Returns the number of particles with nonzero probability.
    """
    return np.sum(self._weights > 0)

  def sample_no_constraints(self):
    """
    Generate a permutation that maps true values (indices) to their new value.
    """
    # Reversed list from 12 to 0.
    orig_perm = list(range(13))[::-1]
    prop_perm = []

    # Random offsets to pop from.
    seed = np.random.geometric(p=0.25, size=13) - 1
    for s in seed:
      # Because pop operates from the back of the list, need to subtract length of list to start
      # popping at the beginning.
      pop_i = len(orig_perm) - 1 - (s % len(orig_perm))
      prop_perm.append(orig_perm.pop(pop_i))

    return Permutation(prop_perm)

  def sample_with_constraints(self):
    # Reversed list from 12 to 0.
    # orig_perm = list(range(13))[::-1]
    # prop_perm = []

    # # Random offsets to pop from.
    # seed = np.random.geometric(p=0.25, size=13) - 1
    # for s in seed:
    #   pop_i = len(orig_perm) - 1 - (s % len(orig_perm))
    #   prop_perm.append(orig_perm.pop(pop_i))
    # return prop_perm
    raise NotImplementedError()
