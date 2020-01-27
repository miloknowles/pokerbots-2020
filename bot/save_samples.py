from permutation_filter import *
import numpy as np


if __name__ == "__main__":
  N = int(1e7)

  perms = np.zeros((N, 13), dtype=np.uint8)

  pf = PermutationFilter(1)
  for i in range(N):
    if i % 1e5 == 0:
      print("Finished {} percent of samples".format(i / N))
    p = pf.sample_uniform()
    perms[i] = p.perm_to_true

  np.save("samples.npy", perms)
