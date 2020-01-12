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
    self.true_to_perm = np.zeros(13).astype(int)

    for perm_val, true_val in enumerate(self.perm_to_true):
      self.true_to_perm[true_val] = perm_val

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


# def is_consistent_with_result(p, result):
#   board_and_winner_true = [eval7.Card(c) for c in p.map_str(result.board_cards + result.winner_hole_cards)]
#   board_and_loser_true = [eval7.Card(c) for c in p.map_str(result.board_cards + result.loser_hole_cards)]

#   score_winner = eval7.evaluate(board_and_winner_true)
#   score_loser = eval7.evaluate(board_and_loser_true)

#   # If the losing hand has a higher score than the winning score, then the permutation is
#   # inconsistent with observations.
#   if score_winner <= score_loser:
#     return False
#   else:
#     return True


# def satisfies_all_results(p, results):
#   for r in results:
#     if not is_consistent_with_result(p, r):
#       return False
#   return True

# def valid_permutation_recursive(perm_to_true, i, results, remaining_values):
#   """
#   Return a permutation that satisfies all of the results so far.

#   perm_to_true: The permutation so far (valid up until p[i])
#   """
#   # If i >= len(perm_to_true), then we've filled in everything without violating constraints.
#   if i >= len(perm_to_true):
#     return True, perm_to_true

#   # Otherwise, try every option for perm_to_true[i].
#   for v in remaining_values:
#     is_valid, valid_perm_to_true = valid_
  
#   if not self.satisfies_all_results()

#   return False, None


class PermutationFilter(object):
  def __init__(self, num_particles):
    self._num_particles = num_particles

    self._particles = [self.sample_uniform() for _ in range(self._num_particles)]
    self._weights = np.ones(self._num_particles) / self._num_particles

    self._results = []
    self._lifeline = None

    self._cache = np.load("./samples.npy")
    print(self._cache)

  def update(self, result):
    for i, p in enumerate(self._particles):
      if self._weights[i] > 0:
        if not self.is_consistent_with_result(p, result):
          self._weights[i] = 0
        else:
          self._lifeline = p
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

  def resample_valid(self, invalid):
    """
    Quickly mutate an invalid permutation until it satisfies result.
    """
    fixed = invalid.perm_to_true.copy()
    while not self.satisfies_all_results(Permutation(fixed)):
      ij = np.random.choice(13, size=2, replace=False)
      i, j = ij[0], ij[1]
      oi, oj = fixed[i], fixed[j]
      fixed[i] = oj
      fixed[j] = oi 
    return Permutation(fixed)

  def resample(self, nparticles):
    # If we have no valid particles right now, need to do some expensive work to get a valid one.
    if self.nonzero() == 0:
      print("WARNING: EXPENSIVE RESAMPLE")
      # t0 = time.time()
      # self._particles[0] = self.resample_valid(self._lifeline)
      # self._weights[0] = 1
      # elapsed = time.time() - t0
      # print("Took {} sec to regenerate a valid perm".format(elapsed))
      for i in range(len(self._cache)):
        p = Permutation(self._cache[i])
        if self.satisfies_all_results(p):
          self._particles[0] = p
          self._weights[0] = 1
          did_get_valid_particle = True

      if did_get_valid_particle:
        print("Found valid sample in the cache")
      else:
        print("Could not find valid in cache, randomly sampling")

      did_get_valid_particle = False
      while not did_get_valid_particle:
        p = self.sample_uniform()
        is_valid = True
        for r in self._results:
          if not self.is_consistent_with_result(p, r):
            is_valid = False
            break
        if is_valid:
          self._particles[0] = p
          self._weights[0] = 1
          did_get_valid_particle = True
    
    # If we do have some valid particles, randomly choose one and use for MCMC.
    valid_particles = []
    for i in self._weights.nonzero()[0][:nparticles]:
      valid_particles.append(self._particles[i])
    self._particles = valid_particles

    # NOTE(milo): Number of particles is updated here! We may want to resample fewer than the
    # original number.
    self._num_particles = nparticles

    original_pop = len(self._particles)

    while len(self._particles) < self._num_particles:
      # original_perm = self._particles[np.random.choice(len(self._particles))]
      original_perm = self._particles[np.random.choice(original_pop)]

      # Try a few extra times to get a new sample.
      did_accept = False
      for i in range(1):
        p, did_accept = self.sample_mcmc(original_perm)
        if did_accept:
          break
      self._particles.append(p)

    self._weights = np.ones(self._num_particles) / self._num_particles

  def nonzero(self):
    """
    Returns the number of particles with nonzero probability.
    """
    return np.sum(self._weights > 0)

  def unique(self):
    unique_set = set()
    for p in self._particles:
      s = "".join([str(c) for c in p.true_to_perm])
      unique_set.add(s)
    return len(unique_set)

  def get_unique_permutations(self):
    unique = []

    unique_set = set()
    for p in self._particles:
      s = "".join([str(c) for c in p.true_to_perm])
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

  def permute(self, p):
    orig_perm = list(p.perm_to_true)
    orig_perm.reverse()
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
    # TODO(milo): can make this shorter
    oi = proposal[i]
    oj = proposal[j]
    proposal[i] = oj
    proposal[j] = oi
    return Permutation(proposal)

  def sample_mcmc(self, original_perm):
    """
    Samples a new permutation from the posterior distribution given by observed results so far.
    """
    proposal_perm = self.make_proposal_geometric(original_perm)

    # Check if the proposal satisfies all of the constraints from events seen so far.
    is_valid = True
    for r in self._results:
      if not self.is_consistent_with_result(proposal_perm, r):
        is_valid = False
        break

    if is_valid:
      prior_proposal = self.compute_prior(proposal_perm)
      prior_original= self.compute_prior(original_perm)

      # Accept according to Metropolis-Hastings.
      A_ij = min(1, prior_proposal / prior_original)
      if random.random() < A_ij:
        return proposal_perm, True

    return original_perm, False
