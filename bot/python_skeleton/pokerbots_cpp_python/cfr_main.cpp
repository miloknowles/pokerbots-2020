#include <random>

#include "cfr.hpp"
#include "engine_modified.hpp"
#include "pbots_calc.h"

namespace pb {


static std::string ConvertCodeToCard(const int code) {
  std::string out = "xx";
  out[0] = utils::RANK_VAL_TO_STR[code / 4];
  out[1] = utils::SUIT_VAL_TO_STR[code % 4];
  return out;
}

RoundState CreateNewRound(int sb_plyr_idx) {
  std::vector<int> deck;
  for (int i = 0; i < 52; ++i) { deck.emplace_back(i); }
  std::random_shuffle(deck.begin(), deck.end());

  std::array<std::array<std::string, 2>, 2> hands;
  std::array<std::string, 5> board;

  hands[0][0] = ConvertCodeToCard(deck[0]);
  hands[0][1] = ConvertCodeToCard(deck[1]);

  hands[1][0] = ConvertCodeToCard(deck[2]);
  hands[1][1] = ConvertCodeToCard(deck[3]);

  board[0] = ConvertCodeToCard(deck[4]);
  board[1] = ConvertCodeToCard(deck[5]);
  board[2] = ConvertCodeToCard(deck[6]);
  board[3] = ConvertCodeToCard(deck[7]);
  board[4] = ConvertCodeToCard(deck[8]);

  std::array<int, 2> pips = { 1, 2 };
  std::array<int, 2> stacks = { 199, 198 };

  if (sb_plyr_idx == 1) {
    pips = { pips[1], pips[0] };
    stacks = { stacks[1], stacks[0] };
  }

  return RoundState(sb_plyr_idx, 0, pips, stacks, hands, board, nullptr, {{1, 2}}, sb_plyr_idx);
}


static float PbotsCalcEquity(
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


PrecomputedEv MakePrecomputedEv(const RoundState& round_state) {
  PrecomputedEv out; // 2x5

  const std::string h1 = round_state.hands[0][0] + round_state.hands[0][1];
  const std::string h2 = round_state.hands[1][0] + round_state.hands[1][1];

  for (int s = 0; s < 4; ++s) {
    int iters = 1;
    if (s == 1) {
      iters = 10000;
    } else if (s == 2) {
      iters = 10000;
    } else if (s == 3) {
      iters = 1326;
    }

    const std::string board = round_state.deck[0] + round_state.deck[1] + round_state.deck[2] + round_state.deck[3] + round_state.deck[4];

    const float ev1 = PbotsCalcEquity(h1 + ":xx", board, "", iters);
    const float ev2 = PbotsCalcEquity(h2 + ":xx", board, "", iters);
    out[0][s] = ev1;
    out[1][s] = ev2;
  }
}


void DoCfrIterationForPlayer(int t, int num_traversals, int traverse_plyr) {
  std::array<RegretMatchedStrategy, 2> regrets;
  std::array<RegretMatchedStrategy, 2> strategies;

  for (int k = 0; k < num_traversals; ++k) {
    const int sb_plyr_idx = k % 2;
    RoundState round_state = CreateNewRound(sb_plyr_idx);

    std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
    PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

    int rctr = 0;

    const NodeInfo info = TraverseCfr(
        &round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
        t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

  }

  std::cout << "Done" << std::endl;
}

}

int main(int argc, char const *argv[])
{
  pb::DoCfrIterationForPlayer(0, 1, 0);
  return 0;
}
