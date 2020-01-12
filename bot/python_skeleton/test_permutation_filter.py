import unittest, time

from permutation_filter import *


class PermutationFilterTest(unittest.TestCase):
  def test_construct(self):
    pf = PermutationFilter(1000)
    for p in pf._particles:
      print(p)

  def test_resample(self):
    t0 = time.time()
    pf = PermutationFilter(1000)

    num_iters = 100
    for _ in range(num_iters):
      pf.resample()

    elapsed = time.time() - t0
    print("Took {} sec to resample {} times".format(elapsed, num_iters))

  def test_showdown_result(self):
    r = ShowdownResult(["Ac", "3d"], ["2d", "2s"], ["Js", "Qs", "Ks"])
    print(r)

  def test_prior(self):
    t0 = time.time()
    pf = PermutationFilter(1000)

    for _ in range(10):
      perm = pf.sample_uniform()
      prior = pf.compute_prior(perm)
      print(perm)
      print("Prior={}".format(prior))
  
  def test_sample_mcmc(self):
    # First try with no constraints.
    pf = PermutationFilter(1000)

    t0 = time.time()
    original_perm = pf.sample_uniform()
    nresample = 1000
    accept_ctr = 0
    for _ in range(nresample):
      new_perm = pf.sample_mcmc(original_perm)
      if not (new_perm.true_to_perm == original_perm.true_to_perm).all():
        accept_ctr += 1
    elapsed = time.time() - t0
    accept_frac = accept_ctr / nresample
    print("MCMC (no results) t={} | AF={}".format(elapsed, accept_frac))

    # Now try with a showndown result.
    r = ShowdownResult(["Ac", "3d"], ["2d", "2s"], ["Js", "Qs", "Ks"])
    pf.update(r)

    # Get a particle that is still valid.
    valid_idx = np.random.choice((pf._weights > 0).nonzero()[0])
    original_perm = pf._particles[valid_idx]

    t0 = time.time()
    nresample = 1000
    accept_ctr = 0
    for _ in range(nresample):
      new_perm = pf.sample_mcmc(original_perm)
      if not (new_perm.true_to_perm == original_perm.true_to_perm).all():
        accept_ctr += 1
    elapsed = time.time() - t0
    accept_frac = accept_ctr / nresample
    print("MCMC (1 results) t={} | AF={}".format(elapsed, accept_frac))

  def test_unique(self):
    pf = PermutationFilter(1000)
    print(pf.unique())

  def test_convergence(self):
    results = []
    with open("./showdown_results_01.txt", "r") as f:
      for l in f:
        l = l.replace("\n", "").split(" | ")
        winning_hand = l[0].split(" ")
        losing_hand = l[1].split(" ")
        board_cards = l[2].split(" ")
        results.append(ShowdownResult(winning_hand, losing_hand, board_cards))

    #  2 3 4 5 6 7 8 9 T J Q K A 
    # [6 7 8 3 2 4 9 K T A J 5 Q]
    true_perm = Permutation(np.array([4, 5, 6, 1, 0, 2, 7, 11, 8, 12, 9, 3, 10]))

    pf = PermutationFilter(10000)
    for i, r in enumerate(results):
      pf.update(r)
      has_true_perm = pf.has_particle(true_perm)
      print("iter={} Has true perm? {}".format(i, has_true_perm))

      if pf.nonzero() < 1000:
        pf.resample(1000)
        print("Did resample, now has {} unique".format(pf.unique()))
    
    for un in pf.get_unique_permutations():
      print(un)

if __name__ == "__main__":
  unittest.main()
