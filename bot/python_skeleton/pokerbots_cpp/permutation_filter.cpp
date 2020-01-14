#include "./permutation_filter.hpp"

#include "pbots_calc.h"

#include <poker-eval/enumdefs.h>
#include <poker-eval/poker_defs.h>

namespace pb {

float PbotsCalcEquity(const std::string& query,
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

  // Query pbots_calc.
  calc(query.c_str(), board_c, dead_c, iters, res);
  const float ev = res->ev[0];

  // Free memory after allocating.
  free_results(res);
  delete[] board_c;
  delete[] dead_c;
  return ev;
}

}
