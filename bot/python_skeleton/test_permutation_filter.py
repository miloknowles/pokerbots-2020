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


if __name__ == "__main__":
  unittest.main()
