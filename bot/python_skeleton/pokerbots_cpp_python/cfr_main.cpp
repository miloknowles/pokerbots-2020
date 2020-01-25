#include <random>

#include "cfr.hpp"
#include "engine_modified.hpp"

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

void DoCfrIteration(int t) {
  const int num_traversals = 10;
  const int traverse_plyr = 0;

  std::array<RegretMatchedStrategy, 2> regrets;
  std::array<RegretMatchedStrategy, 2> strategies;

  for (int k = 0; k < num_traversals; ++k) {

    const int sb_plyr_idx = k % 2;

    RoundState* round_state = CreateNewRound(sb_plyr_idx);

    const NodeInfo info = TraverseCfr(
        round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
        t, reach_probabilities, precomputed_ev, &rctr, true, false, false);

  }
}

}

int main(int argc, char const *argv[])
{
  /* code */
  return 0;
}
