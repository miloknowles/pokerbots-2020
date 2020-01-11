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
      perm = pf.sample_no_constraints()
      prior = pf.compute_prior(perm)
      print(perm)
      print("Prior={}".format(prior))

if __name__ == "__main__":
  unittest.main()
