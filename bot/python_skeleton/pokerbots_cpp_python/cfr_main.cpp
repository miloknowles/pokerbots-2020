#include <random>
#include <ctime>

#include "cfr.hpp"
#include "engine_modified.hpp"

namespace pb {
namespace cfr {

void DoCfrIterationForPlayer(int t, int num_traversals, int traverse_plyr) {
  std::array<RegretMatchedStrategy, 2> regrets;
  std::array<RegretMatchedStrategy, 2> strategies;

  for (int k = 0; k < num_traversals; ++k) {
    const int sb_plyr_idx = k % 2;
    RoundState round_state = CreateNewRound(sb_plyr_idx);

    std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
    PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

    int rctr = 0;

    std::cout << round_state.hands[0][0] << round_state.hands[0][1] << std::endl;
    std::cout << round_state.hands[1][0] << round_state.hands[1][1] << std::endl;

    for (int i = 0; i < 5; ++i) {
      std::cout << round_state.deck[i] << " ";
    }
    std::cout << std::endl;

    const NodeInfo info = TraverseCfr(
        &round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
        t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

    printf("Tree size = %d | r0=%d r1=%d s0=%d s1=%d\n", rctr, regrets[0].Size(), regrets[1].Size(),
      strategies[0].Size(), strategies[1].Size());

    printf("exploit=[%f %f]\n", info.exploitability[0], info.exploitability[1]);
  }

  std::cout << "Done" << std::endl;
}

}
}

int main(int argc, char const *argv[]) {
  std::srand(std::time(0));
  pb::cfr::DoCfrIterationForPlayer(0, 10, 0);
  return 0;
}
