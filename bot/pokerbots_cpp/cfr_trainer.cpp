#include "cfr_trainer.hpp"

namespace pb {
namespace cfr {

NodeInfo DoCfrIterationForPlayer(std::array<RegretMatchedStrategyKmeans, 2>& regrets,
                             std::array<RegretMatchedStrategyKmeans, 2>& strategies,
                             int t, int traverse_plyr, const Options& opt,
                             bool debug_print) {
  NodeInfo info;

  for (int k = 0; k < 2; ++k) {
    const int sb_plyr_idx = k % 2;
    RoundState round_state = CreateNewRound(sb_plyr_idx);

    std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
    PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

    int rctr = 0;

    info = TraverseCfr(
        round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
        t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

    if (debug_print) {
      printf("[TRAVERSE] treesize=%d | exploit=[%f %f] | r0=%d r1=%d s0=%d s1=%d | \n",
        rctr, info.exploitability[0], info.exploitability[1], regrets[0].Size(), regrets[1].Size(),
        strategies[0].Size(), strategies[1].Size());

      // std::cout << round_state.hands[0][0] << round_state.hands[0][1] << std::endl;
      // std::cout << round_state.hands[1][0] << round_state.hands[1][1] << std::endl;
      // for (int i = 0; i < 5; ++i) {
      //   std::cout << round_state.deck[i] << " ";
      // }
      // std::cout << std::endl;
    }
  }

  return info;
}

}
}
