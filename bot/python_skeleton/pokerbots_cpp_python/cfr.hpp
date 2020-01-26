#pragma once

// #include <map>
#include <cassert>
#include <algorithm>

#include "infoset.hpp"
#include "pbots_calc.h"
#include "engine_modified.hpp"
#include "regret_matched_strategy.hpp"

namespace pb {
namespace cfr {


std::pair<ActionVec, ActionMask> MakeActions(RoundState* round_state, int active);


EvInfoSet MakeInfoSet(const RoundState* round_state, int active_plyr_idx, bool player_is_sb,
                      PrecomputedEv precomputed_ev);


inline std::string ConvertCodeToCard(const int code) {
  std::string out = "xx";
  out[0] = utils::RANK_VAL_TO_STR[code / 4];
  out[1] = utils::SUIT_VAL_TO_STR[code % 4];
  return out;
}


RoundState CreateNewRound(int sb_plyr_idx);


ActionRegrets ApplyMaskAndUniform(const ActionRegrets& p, const ActionMask& mask);


PrecomputedEv MakePrecomputedEv(const RoundState& round_state);


inline float PbotsCalcEquity(
    const std::string& query,
    const std::string& board,
    const std::string& dead,
    const size_t iters) {
  Results* res = alloc_results();

  // Need to convert board and dead to mutable char* type.
  char* board_c = new char[board.size() + 1];
  char* dead_c = new char[dead.size() + 1];
  std::copy(board.begin(), board.end(), board_c);
  std::copy(dead.begin(), dead.end(), dead_c);
  board_c[board.size()] = '\0';
  dead_c[dead.size()] = '\0';

  char* query_c = new char[query.size() + 1];
  std::copy(query.begin(), query.end(), query_c);
  query_c[query.size()] = '\0';

  // Query pbots_calc.
  calc(query_c, board_c, dead_c, iters, res);
  const float ev = res->ev[0];

  // Free memory after allocating.
  free_results(res);
  delete[] board_c;
  delete[] dead_c;
  return ev;
}


struct NodeInfo {
  NodeInfo() {
    std::fill(strategy_ev.begin(), strategy_ev.end(), 0);
    std::fill(best_response_ev.begin(), best_response_ev.end(), 0);
    std::fill(exploitability.begin(), exploitability.end(), 0);
  }

  std::array<double, 2> strategy_ev;
  std::array<double, 2> best_response_ev;
  std::array<double, 2> exploitability; 
};


NodeInfo TraverseCfr(State* state,
                     int traverse_plyr,
                     int sb_plyr_idx,
                     std::array<RegretMatchedStrategy, 2>& regrets,
                     std::array<RegretMatchedStrategy, 2>& strategies,
                     int t,
                     const std::array<double, 2>& reach_probabilities,
                     const PrecomputedEv& precomputed_ev,
                     int* rctr,
                     bool allow_updates = true,
                     bool do_external_sampling = false,
                     bool skip_unreachable_actions = false);

}
}
