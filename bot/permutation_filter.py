import numpy as np
import eval7
from collections import OrderedDict
import random, time


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
  def __init__(self, perm_to_true):
    """
    Defines a two-way mapping between true values and permutation values (0-13). The zeroth index
    represents a 2, and the 13th index represents an A.

    2 3 4 5 6 7 8 9 T J Q  K  A
    0 1 2 3 4 5 6 7 8 9 10 11 12

    perm_to_true (list of int) : Should have length 13. perm_to_true[i] = j means that the ith value
                                 that we see corresponds to a true value of j.
    """
    self.perm_to_true = np.array(perm_to_true).astype(int)

  def map_str(self, cards_str):
    """
    Maps permuted card values to their true values.
    """
    out = []
    for c in cards_str:
      rank, suit = str_to_rank_and_suit(c)
      out.append(rank_and_suit_to_str(self.perm_to_true[rank], suit))
    return out

  def __str__(self):
    cards = ["{}x".format(v) for v in RANKS_ORDERED]
    perm = self.map_str(cards)
    return ' ' + ' '.join(cards) + ' ' + '\n' + '[' + ' '.join(perm) + ']'


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

  def mapped_result(self, p):
    return "Winning hand: " + " ".join(p.map_str(self.winner_hole_cards)) + " | " \
           "Losing hand: " + " ".join(p.map_str(self.loser_hole_cards)) + " | " \
           "Board cards: " + " ".join(p.map_str(self.board_cards))

  def get_card_values(self):
    return ([RANK_STR_TO_VAL[c[0]] for c in self.winner_hole_cards],
            [RANK_STR_TO_VAL[c[0]] for c in self.loser_hole_cards],
            [RANK_STR_TO_VAL[c[0]] for c in self.board_cards])


class PermutationFilter(object):
  def __init__(self, num_particles):
    self._num_particles = num_particles

    self._particles = [self.sample_uniform() for _ in range(self._num_particles)]
    self._weights = np.ones(self._num_particles) / self._num_particles

    self._results = []
    self._dead_indices = []

    self._invalid_tries = 0
    self._invalid_retry_success = 0

  def invalid_retry_success_rate(self):
    return self._invalid_retry_success / self._invalid_tries

  def update(self, result):
    nonzero = self.nonzero()

    num_invalid_retries = min(int(5 * self._num_particles / nonzero), 10)
    # num_valid_retries = 2 if nonzero < (0.5 * self._num_particles) else 1
    num_valid_retries = 2
    print("Retries: {} {}".format(num_invalid_retries, num_valid_retries))

    for i, p in enumerate(self._particles):
      # Only update particles that are still alive.
      if self._weights[i] > 0:
        if not self.is_consistent_with_result(p, result):
          # Particle died on this iteration, use the invalid resampling procedure.
          for _ in range(num_invalid_retries):
            self._invalid_tries += 1
            proposal, is_valid = self.sample_mcmc_invalid(p, result)
            if is_valid:
              self._weights[i] = 1
              self._particles[i] = proposal
              self._invalid_retry_success += 1
              break
          
          if not is_valid:
            self._weights[i] = 0
            self._dead_indices.append(i)

        # If the particle is alive after update, make some samples from it.
        else:
          for _ in range(num_valid_retries):
            if len(self._dead_indices) == 0:
              break
            proposal, is_valid = self.sample_mcmc_valid(p, result)
            if is_valid:
              dead_index_to_replace = self._dead_indices.pop()
              self._weights[dead_index_to_replace] = 1
              self._particles[dead_index_to_replace] = proposal
              break

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

  def satisfies_all_results(self, p):
    for r in self._results:
      if not self.is_consistent_with_result(p, r):
        return False
    return True

  def nonzero(self):
    """
    Returns the number of particles with nonzero probability.
    """
    return np.sum(self._weights > 0)

  def unique(self):
    unique_set = set()
    for i, p in enumerate(self._particles):
      if self._weights[i] > 0:
        s = "".join([str(c) for c in p.perm_to_true])
        unique_set.add(s)
    return len(unique_set)

  def get_unique_permutations(self):
    unique = []
    unique_set = set()
    for i, p in enumerate(self._particles):
      if self._weights[i] > 0:
        s = "".join([str(c) for c in p.perm_to_true])
        if s not in unique_set:
          unique.append(p)
          unique_set.add(s)
    return unique

  def has_particle(self, p):
    for other in self._particles:
      if (other.perm_to_true == p.perm_to_true).all():
        return True
    return False

  def sample_uniform(self):
    """
    Generate a permutation of the true card values.
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

  def compute_prior(self, perm):
    """
    Computes the prior probability of a permutation by reverse engineering the indices that must
    have been sampled from a geometric distribution.
    """
    p = 1.0

    # These are the remaining "true" values to be chosen by each permuted value.
    remaining = list(range(13))[::-1]

    for perm_val, true_val in enumerate(perm.perm_to_true):
      pop_i = remaining.index(true_val)
      s = -1 * (pop_i - len(remaining) + 1)
      remaining.pop(pop_i)

      # Since indices wrap around, there are many possible ways we could've popped the value that we
      # did. Approximate the probability by assuming that we don't wrap around more than 3 times.
      p1 = 0.25 * (0.75 ** s)
      p2 = 0.25 * (0.75 ** (s + len(remaining)))
      p3 = 0.25 * (0.75 ** (s + 2*len(remaining)))
      p4 = 0.25 * (0.75 ** (s + 3*len(remaining)))
      p5 = 0.25 * (0.75 ** (s + 4*len(remaining)))
      p *= (p1 + p2 + p3 + p4 + p5)

    return p

  def make_proposal(self, p):
    nswaps = 1
    proposal = p.perm_to_true.copy()

    for _ in range(nswaps):
      ij = np.random.choice(13, size=2, replace=False)
      i, j = ij[0], ij[1]

      # Swap the values at i and j to make a candidate.
      oi = proposal[i]
      oj = proposal[j]
      proposal[i] = oj
      proposal[j] = oi

    return Permutation(proposal)

  def make_proposal_geometric(self, p):
    i = np.random.choice(13)
    j = (i + np.random.geometric(0.25)) % 13
    proposal = p.perm_to_true.copy()
    oi = proposal[i]
    oj = proposal[j]
    proposal[i] = oj
    proposal[j] = oi
    return Permutation(proposal)

  def make_proposal_from_invalid(self, p, winner_vals, loser_vals, other_vals):
    """
    Make a proposal permutation by swapping a card from the winner's hand, loser's hand, or
    remaining deck.
    """
    i = random.randint(0, 1)
    j = random.randint(0, len(loser_vals) + len(other_vals) - 1)

    if random.random() < 0.5:
      vi = winner_vals[i]
      vj = (loser_vals + other_vals)[j]
    else:
      vi = loser_vals[i]
      vj = (winner_vals + other_vals)[j]

    proposal = p.perm_to_true.copy()
    ti, tj = proposal[vi], proposal[vj]
    proposal[vi] = tj
    proposal[vj] = ti

    return Permutation(proposal)
  
  def make_proposal_from_valid(self, p, winner_vals, loser_vals, board_vals):
    i = np.random.choice(13)
    if i in winner_vals:
      vj = winner_vals[(winner_vals.index(i) + 1) % 2]
    elif i in loser_vals:
      vj = loser_vals[(loser_vals.index(i) + 1) % 2]
    elif i in board_vals:
      vj = board_vals[np.random.choice(len(board_vals))]
    else:
      other_vals = np.ones(13)
      for vals_to_remove in (winner_vals, loser_vals, board_vals):
        other_vals[vals_to_remove] = 0
      other_vals = other_vals.nonzero()[0]
      vj = np.random.choice(other_vals)

    proposal = p.perm_to_true.copy()
    ti, tj = p.perm_to_true[i], p.perm_to_true[vj]
    proposal[i] = tj
    proposal[vj] = ti

    return Permutation(proposal)

  def metropolis_hastings(self, original_perm, proposal_perm):
    prior_proposal = self.compute_prior(proposal_perm)
    prior_original = self.compute_prior(original_perm)

    # Do the A_ij check first to avoid the expensive call to check all results.
    A_ij = min(1, prior_proposal / prior_original)
    if random.random() < A_ij:
      if self.satisfies_all_results(proposal_perm):
        return proposal_perm, True

    return original_perm, False

  def sample_mcmc_invalid(self, original_perm, result):
    winner_vals, loser_vals, board_vals = result.get_card_values()
    other_vals = np.ones(13)
    for vals_to_remove in (winner_vals, loser_vals, board_vals):
      other_vals[vals_to_remove] = 0
    other_vals = other_vals.nonzero()[0]

    proposal_perm = self.make_proposal_from_invalid(original_perm, winner_vals, loser_vals, list(other_vals))
    return self.metropolis_hastings(original_perm, proposal_perm)

  def sample_mcmc_valid(self, original_perm, result):
    """
    Swap cards in either hand or on the board.
    """
    winner_vals, loser_vals, board_vals = result.get_card_values()

    proposal_perm = self.make_proposal_from_valid(original_perm, winner_vals, loser_vals, board_vals)
    return self.metropolis_hastings(original_perm, proposal_perm)
