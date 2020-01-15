from __future__ import print_function
from pokerbots_cpp_python import pokerbots_cpp_python as cpp

# https://github.com/tensorflow/tensorflow/issues/6968
def test_converge_cpp():
  results = []
  with open("./showdown_results_02.txt", "r") as f:
    for l in f:
      if "|" not in l:
        continue
      l = l.replace("\n", "").split(" | ")
      winning_hand = l[0].split(" ")
      losing_hand = l[1].split(" ")
      board_cards = l[2].split(" ")

      for c in winning_hand:
        assert(c not in losing_hand)
        assert(c not in board_cards)
      for c in losing_hand:
        assert(c not in board_cards)

      results.append(cpp.ShowdownResult("".join(winning_hand), "".join(losing_hand), "".join(board_cards)))

  print("Replaying %d results" % len(results))

  #  2 3 4 5 6 7 8 9 T J Q K A 
  # [3 4 9 2 8 Q K A 5 6 J T 7]
  # true_perm = Permutation(np.array([1, 2, 7, 0, 6, 10, 11, 12, 3, 4, 9, 8, 5]))
  true_perm_vl = cpp.ValueList()
  for v in [1, 2, 7, 0, 6, 10, 11, 12, 3, 4, 9, 8, 5]:
    true_perm_vl.append(v)

  pf = cpp.PermutationFilter(10000)

  # Do everything but the last result.
  # prev_unique_particles = None

  for i, r in enumerate(results):
    pf.Update(r)
    print("%d: Filter has %s nonzero" % (i, pf.Nonzero()))

    if pf.HasPermutation(true_perm_vl):
      print("Found true permutation!")
      break
    # if (i > 100):
    #   print("Did 100 showdowns, assume converged")
    #   break
  pf.Profile()
 
if __name__ == "__main__":
  test_converge_cpp()
