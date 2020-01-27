#include <random>
#include <ctime>
#include <fstream>
#include <boost/filesystem.hpp>

#include "cfr.hpp"
#include "engine_modified.hpp"

namespace pb {
namespace cfr {

namespace fs = boost::filesystem;


struct Options {
  std::string EXPERIMENT_NAME = "MC_CFR_01";
  std::string EXPERIMENT_PATH = "/home/milo/pokerbots-2020/cfr/" + EXPERIMENT_NAME + "/";

  int NUM_CFR_ITERS = 1000;
  int NUM_TRAVERSALS_PER_ITER = 10;
  int NUM_TRAVERSALS_EVAL =  10;
  int TRAVERSAL_PRINT_HZ = 5; 
};


void DoCfrIterationForPlayer(std::array<RegretMatchedStrategy, 2>& regrets,
                             std::array<RegretMatchedStrategy, 2>& strategies,
                             int t, int traverse_plyr, const Options& opt) {
  for (int k = 0; k < opt.NUM_TRAVERSALS_PER_ITER; ++k) {
    const int sb_plyr_idx = k % 2;
    RoundState round_state = CreateNewRound(sb_plyr_idx);

    std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
    PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

    int rctr = 0;

    const NodeInfo info = TraverseCfr(
        round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
        t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

    if (k % opt.TRAVERSAL_PRINT_HZ == 0) {
      printf("[TRAVERSE] treesize=%d | exploit=[%f %f] | r0=%d r1=%d s0=%d s1=%d | \n",
          rctr, info.exploitability[0], info.exploitability[1], regrets[0].Size(), regrets[1].Size(),
          strategies[0].Size(), strategies[1].Size());

      // Print hands and deck.
      std::cout << round_state.hands[0][0] << round_state.hands[0][1] << std::endl;
      std::cout << round_state.hands[1][0] << round_state.hands[1][1] << std::endl;
      for (int i = 0; i < 5; ++i) {
        std::cout << round_state.deck[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  printf("Done with %d traversals\n", opt.NUM_TRAVERSALS_PER_ITER);
}


class CfrTrainer {
 public:
  CfrTrainer(const Options& opt) : opt_(opt) {
    std::cout << "EXPERIMENT_NAME: " << opt.EXPERIMENT_NAME << std::endl;
    std::cout << "EXPERIMENT_PATH: " << opt.EXPERIMENT_PATH << std::endl;

    const fs::path dir(opt.EXPERIMENT_PATH.c_str());
    if (boost::filesystem::create_directory(dir)) {
      std::cout << "NOTE: EXPERIMENT_PATH didn't exist, created it" << std::endl;
    }

    // Save everything once to have empty files.
    const std::string test_path = opt.EXPERIMENT_PATH + "total_regrets_0.txt";
    if (!fs::exists(test_path)) {
      regrets_[0].Save(opt.EXPERIMENT_PATH + "total_regrets_0.txt");
      regrets_[1].Save(opt.EXPERIMENT_PATH + "total_regrets_1.txt");
      strategies_[0].Save(opt.EXPERIMENT_PATH + "avg_strategy_0.txt");
      strategies_[1].Save(opt.EXPERIMENT_PATH + "avg_strategy_1.txt");
    } else {
      std::cout << "WARNING: Files already existed on disk, resuming!" << std::endl;
    }

    regrets_[0].Load(opt.EXPERIMENT_PATH + "total_regrets_0.txt");
    regrets_[1].Load(opt.EXPERIMENT_PATH + "total_regrets_1.txt");
    strategies_[0].Load(opt.EXPERIMENT_PATH + "avg_strategy_0.txt");
    strategies_[1].Load(opt.EXPERIMENT_PATH + "avg_strategy_1.txt");
  }

  void Run() {
    std::srand(rand() % 100);
    // std::time:
  
    for (int t = 0; t < opt_.NUM_CFR_ITERS; ++t) {
      // Do some traversals for each player.
      for (int tp = 0; tp < 2; ++tp) {
        DoCfrIterationForPlayer(regrets_, strategies_, t, tp, opt_);
      }

      // Save everything for this iteration.
      regrets_[0].Save(opt_.EXPERIMENT_PATH + "total_regrets_0.txt");
      regrets_[1].Save(opt_.EXPERIMENT_PATH + "total_regrets_1.txt");
      strategies_[0].Save(opt_.EXPERIMENT_PATH + "avg_strategy_0.txt");
      strategies_[1].Save(opt_.EXPERIMENT_PATH + "avg_strategy_1.txt");

      // Log status to disk.
      EvalAndLog(t);
    }
  }

  void EvalAndLog(int t) {
    std::ofstream fout;
    fout.open(opt_.EXPERIMENT_PATH + "./logs.txt", std::ios::app);
    assert(fout.is_open());
    fout << "*** EVALUATING t=" << t << std::endl;
    fout.close();

    // Run evaluation on the average strategy.
  //   for (int k = 0; k < opt.NUM_TRAVERSALS_PER_ITER; ++k) {
  //   const int sb_plyr_idx = k % 2;
  //   RoundState round_state = CreateNewRound(sb_plyr_idx);

  //   std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
  //   PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

  //   int rctr = 0;

  //   const NodeInfo info = TraverseCfr(
  //       &round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
  //       t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

  //   if (k % opt.TRAVERSAL_PRINT_HZ == 0) {
  //     printf("[TRAVERSE] treesize=%d | exploit=[%f %f] | r0=%d r1=%d s0=%d s1=%d | \n",
  //         rctr, info.exploitability[0], info.exploitability[1], regrets[0].Size(), regrets[1].Size(),
  //         strategies[0].Size(), strategies[1].Size());
  //     // std::cout << round_state.hands[0][0] << round_state.hands[0][1] << std::endl;
  //     // std::cout << round_state.hands[1][0] << round_state.hands[1][1] << std::endl;

  //     // for (int i = 0; i < 5; ++i) {
  //       // std::cout << round_state.deck[i] << " ";
  //     // }
  //     // std::cout << std::endl;
  //   }
  // }
  }

 private:
  Options opt_;

  std::array<RegretMatchedStrategy, 2> regrets_;
  std::array<RegretMatchedStrategy, 2> strategies_;
};

}
}

int main(int argc, char const *argv[]) {
  pb::cfr::Options opt;
  pb::cfr::CfrTrainer trainer(opt);
  trainer.Run();
  std::cout << "Finished running CFR" << std::endl;
  return 0;
}
