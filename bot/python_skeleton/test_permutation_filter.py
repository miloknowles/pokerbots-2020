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
      pf.resample(1000)

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
      new_perm, did_accept = pf.sample_mcmc(original_perm)
      if not (new_perm.perm_to_true == original_perm.perm_to_true).all():
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
      new_perm, did_accept = pf.sample_mcmc(original_perm)
      if not (new_perm.perm_to_true == original_perm.perm_to_true).all():
        accept_ctr += 1
    elapsed = time.time() - t0
    accept_frac = accept_ctr / nresample
    print("MCMC (1 results) t={} | AF={}".format(elapsed, accept_frac))

  def test_unique(self):
    pf = PermutationFilter(1000)
    print(pf.unique())

  def test_convergence(self):
    results = []
    with open("./showdown_results_02.txt", "r") as f:
      for l in f:
        if "|" not in l:
          continue
        l = l.replace("\n", "").split(" | ")
        winning_hand = l[0].split(" ")
        losing_hand = l[1].split(" ")
        board_cards = l[2].split(" ")
        results.append(ShowdownResult(winning_hand, losing_hand, board_cards))
    print("Replaying {} results".format(len(results)))

    #  2 3 4 5 6 7 8 9 T J Q K A 
    # [6 7 8 3 2 4 9 K T A J 5 Q]
    # true_perm = Permutation(np.array([4, 5, 6, 1, 0, 2, 7, 11, 8, 12, 9, 3, 10]))

    #  2 3 4 5 6 7 8 9 T J Q K A 
    # [3 4 9 2 8 Q K A 5 6 J T 7]
    true_perm = Permutation(np.array([1, 2, 7, 0, 6, 10, 11, 12, 3, 4, 9, 8, 5]))

    pf = PermutationFilter(5000)

    # Do everything but the last result.
    prev_unique_particles = None

    for i, r in enumerate(results[:-1]):
      pf.update(r)
      has_true_perm = pf.has_particle(true_perm)
      print("iter={} Has true perm? {} nonzero={} invalid_succ_rate={}".format(
        i, has_true_perm, pf.nonzero(), pf.invalid_retry_success_rate()))
      # print("Updated, particle now has {} unique".format(pf.unique()))

      curr_unique_particles = pf.get_unique_permutations()
      if len(curr_unique_particles) == 0 and len(prev_unique_particles) > 0:
        print("================ Last alive particles died off ===============")
        print(r.mapped_result(true_perm))

        print("\nTrue permutation:")
        print(true_perm)
        print("\n")

        for ii, p in enumerate(prev_unique_particles):
          print("Possible permutation {}".format(ii))
          print(p)
          print("Gives result:")
          print(r.mapped_result(p))
          print("\n")
        break

      prev_unique_particles = pf.get_unique_permutations()

if __name__ == "__main__":
  unittest.main()
