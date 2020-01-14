#pragma once

#include <vector>
#include <map>
#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <random>

namespace pb {

typedef std::array<uint8_t, 13> Permutation;


std::map<char, uint8_t> RANK_STR_TO_VAL = {
  {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3}, {'6', 4}, {'7', 5}, {'8', 6},
  {'9', 7}, {'T', 8}, {'J', 9}, {'Q', 10}, {'K', 11}, {'A', 12}
};

std::map<uint8_t, char> RANK_VAL_TO_STR = {
  {0, '2'}, {1, '3'}, {2, '4'}, {3, '5'}, {4, '6'}, {5, '7'}, {6, '8'},
  {7, '9'}, {8, 'T'}, {9, 'J'}, {10, 'Q'}, {11, 'K'}, {12, 'A'}
};


/**
 * @brief Wrapper around pbots_calc library to remove some boilerplate code.
 */
float PbotsCalcEquity(const std::string& query,
                     const std::string& board,
                     const std::string& dead,
                     const size_t iters = 10000);


// Maps a vector of card values to their true values, as defined by permutation p.
inline std::vector<uint8_t> MapToTrueValues(const Permutation& p, const std::vector<uint8_t>& values) {
  std::vector<uint8_t> values_mapped(values.size());
  for (int i = 0; i < values.size(); ++i) {
    values_mapped[i] = p[values[i]];
  }
  return values_mapped;
}


struct ShowdownResult {
 public:
  ShowdownResult(const std::string& winner_hole_cards,
                 const std::string& loser_hole_cards,
                 const std::string& board_cards) :
      winner_hole_cards_(winner_hole_cards),
      loser_hole_cards_(loser_hole_cards),
      board_cards_(board_cards) {
    assert(winner_hole_cards.size() == 4);
    assert(loser_hole_cards.size() == 4);
    assert(board_cards.size() == 10);
  }

  std::vector<uint8_t> GetWinnerValues() const {
    std::vector<uint8_t> out(2);
    out[0] = RANK_STR_TO_VAL[winner_hole_cards_[0]];
    out[1] = RANK_STR_TO_VAL[winner_hole_cards_[1]];
    return out;
  }

  std::vector<uint8_t> GetLoserValues() const {
    std::vector<uint8_t> out(2);
    out[0] = RANK_STR_TO_VAL[loser_hole_cards_[0]];
    out[1] = RANK_STR_TO_VAL[loser_hole_cards_[1]];
    return out;
  }

  std::vector<uint8_t> GetBoardValues() const {
    std::vector<uint8_t> out(5);
    for (int i = 0; i < 10; i += 2) {
      out[i] = RANK_STR_TO_VAL[board_cards_[i]];
    }
    return out;
  }

 private:
  std::string winner_hole_cards_;
  std::string loser_hole_cards_;
  std::string board_cards_;
};


class PermutationFilter {
 public:
  PermutationFilter(int N) : N_(N), particles_(N), weights_(N) {

  }

  // Generate a random permutation from the prior distribution.
  Permutation Sample() {
    std::vector<uint8_t> orig_perm = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<uint8_t> prop_perm;

    std::geometric_distribution<int> distribution(0.25);
    Permutation seed;
    for (int i = 0; i < 13; ++i) {
      seed[i] = distribution(generator_) - 1;
    }

    for (uint8_t s : seed) {
      const int pop_i = s % orig_perm.size();
      prop_perm.emplace_back(orig_perm.at(pop_i));
      orig_perm.erase(orig_perm.begin() + pop_i);
    }

    assert(prop_perm.size() == 13);
    Permutation out;
    for (int i = 0; i < 13; ++i) {
      out[i] = prop_perm[i];
    }
    
    return out;
  }

 private:
  int N_;

  std::vector<Permutation> particles_;
  std::vector<double> weights_;
  std::vector<ShowdownResult> results_ = {};

  std::vector<std::size_t> dead_indices_ = {};

  std::default_random_engine generator_;
};

}
