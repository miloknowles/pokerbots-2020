#pragma once

// TODO: move to source file.
#include <vector>
#include <map>
#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <utility>
#include <set>


namespace pb {

typedef std::array<uint8_t, 13> Permutation;
typedef std::array<uint8_t, 2> HandValues;
typedef std::array<uint8_t, 5> BoardValues;


std::map<char, uint8_t> RANK_STR_TO_VAL = {
  {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3}, {'6', 4}, {'7', 5}, {'8', 6},
  {'9', 7}, {'T', 8}, {'J', 9}, {'Q', 10}, {'K', 11}, {'A', 12}
};

std::map<uint8_t, char> RANK_VAL_TO_STR = {
  {0, '2'}, {1, '3'}, {2, '4'}, {3, '5'}, {4, '6'}, {5, '7'}, {6, '8'},
  {7, '9'}, {8, 'T'}, {9, 'J'}, {10, 'Q'}, {11, 'K'}, {12, 'A'}
};


inline bool Equals(const Permutation& p1, const Permutation& p2) {
  for (int i = 0; i < p1.size(); ++i) {
    if (p1[i] != p2[i]) { return false; }
  }
  return true;
}


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
    const uint8_t mapped = p.at(values.at(i));
    assert(mapped >= 0 && mapped <= 12);
    values_mapped.at(i) = mapped;
  }
  return values_mapped;
}

// Pass in concatenated cards like AcAd4s5h6d.
inline std::string MapToTrueStrings(const Permutation& p, const std::string& strs) {
  std::string out;

  for (int i = 0; i < strs.size(); i += 2) {
    const uint8_t mapped_val = p.at(RANK_STR_TO_VAL.at(strs[i]));
    assert(mapped_val >= 0 && mapped_val <= 12);
    const char mapped_rank = RANK_VAL_TO_STR.at(mapped_val);
    out += (std::string(1, mapped_rank) + std::string(1, strs[i + 1]));
  }
  return out;
}


struct ShowdownResult {
 public:
  ShowdownResult(const std::string& winner_hole_cards,
                 const std::string& loser_hole_cards,
                 const std::string& board_cards) :
      winner_hole_cards(winner_hole_cards),
      loser_hole_cards(loser_hole_cards),
      board_cards(board_cards) {
    assert(winner_hole_cards.size() == 4);
    assert(loser_hole_cards.size() == 4);
    assert(board_cards.size() == 10);
  }

  HandValues GetWinnerValues() const {
    HandValues out;
    out[0] = RANK_STR_TO_VAL[winner_hole_cards[0]];
    out[1] = RANK_STR_TO_VAL[winner_hole_cards[2]];
    assert(out[0] >= 0 && out[1] <= 12);
    return out;
  }

  HandValues GetLoserValues() const {
    HandValues out;
    out[0] = RANK_STR_TO_VAL[loser_hole_cards[0]];
    out[1] = RANK_STR_TO_VAL[loser_hole_cards[2]];
    assert(out[0] >= 0 && out[1] <= 12);
    return out;
  }

  BoardValues GetBoardValues() const {
    BoardValues out;
    for (int i = 0; i < 5; ++i) {
      const uint8_t val = RANK_STR_TO_VAL[board_cards.at(2*i)];
      assert(val >= 0 && val <= 12); 
      out.at(i) = val;
    }
    return out;
  }

  // Public members.
  std::string winner_hole_cards;
  std::string loser_hole_cards;
  std::string board_cards;
};


class PermutationFilter {
 public:
  PermutationFilter(int N) : N_(N), particles_(N), weights_(N), gen_(rd_()) {
    // Sample the initial population of particles.
    for (int i = 0; i < N; ++i) {
      particles_.at(i) = PriorSample();
      weights_.at(i) = 1.0;
    }
  }

  PermutationFilter(const PermutationFilter&) = delete;

  // Generate a random permutation from the prior distribution.
  Permutation PriorSample();

  // Compute the prior probability of sampling Permutation p.
  double ComputePrior(const Permutation& p) const;

  // Does this permutations p satisfy the showdown result r?
  bool SatisfiesResult(const Permutation& p, const ShowdownResult& r) const {
    const std::string& query = MapToTrueStrings(p, r.winner_hole_cards) + ":" + MapToTrueStrings(p, r.loser_hole_cards);
    const std::string& board = MapToTrueStrings(p, r.board_cards);
    const float ev = PbotsCalcEquity(query, board, "", 1);
    return ev > 0;
  }

  // Does permutation p satisfy ALL results seen so far?
  bool SatisfiesAll(const Permutation& p) const {
    for (const ShowdownResult& r : results_) {
      if (!SatisfiesResult(p, r)) { return false; }
    }
    return true;
  }

  int Nonzero() const {
    return (N_ - dead_indices_.size());
  }

  // For a permutation that satisfies all constraints, we don't want to swap values between the
  // hands or the hand and board. Instead, swap cards within hands or within the board.
  Permutation MakeProposalFromValid(const Permutation& p, const ShowdownResult& r);

  // Make a proposal permutation by swapping a card from the winner's hand, loser's hand, or
  // remaining deck.
  Permutation MakeProposalFromInvalid(const Permutation& p, const ShowdownResult& r);

  // Do Metropolis-Hastings rejection sampling on a proposed permutation.
  std::pair<Permutation, bool> MetropolisHastings(const Permutation& orig_perm, const Permutation& prop_perm);

  std::pair<Permutation, bool> SampleMCMCInvalid(const Permutation& orig_perm, const ShowdownResult& r);
  std::pair<Permutation, bool> SampleMCMCValid(const Permutation& orig_perm, const ShowdownResult& r);

  void Update(const ShowdownResult& r);

  bool HasPermutation(const Permutation& query) const {
    for (const Permutation& p : particles_) {
      if (Equals(query, p)) { return true; }
    }
    return false;
  }

  // bool HasPermutationStr(const std::string& str) const {
  //   assert(str.size() == 13);
  //   Permutation parsed;
  //   for (int i = 0; i < 13; ++i) {
  //     parsed.at(i) = static_cast<uint8_t>(std::stoi(std::string(1, str.at(i))));
  //   }
  //   return HasPermutation(parsed);
  // }

 private:
  int N_;

  std::vector<Permutation> particles_;
  std::vector<double> weights_;
  std::vector<ShowdownResult> results_ = {};

  std::vector<int> dead_indices_ = {};

  std::default_random_engine generator_{};
  std::random_device rd_{};
  std::mt19937 gen_;
};


inline std::vector<uint8_t> PermutationToVector(const Permutation& p) {
  std::vector<uint8_t> out;
  for (const uint8_t v : p) {
    out.emplace_back(v);
  }
  return out;
}


inline void PrintPermutation(const Permutation& p) {
  assert(p.size() == 13);
  for (int i = 0; i < p.size(); ++i) {
    std::cout << static_cast<int>(p.at(i)) << " ";
  }
  std::cout << "\n";
}


inline void PrintVector(const std::vector<uint8_t>& vec) {
  for (const uint8_t t : vec) {
    std::cout << static_cast<int>(t) << " ";
  }
  std::cout << "\n";
}


inline bool PermutationIsValid(const Permutation& p) {
  std::set<uint8_t> s;
  for (const uint8_t v : p) {
    if (v < 0 || v > 12) {
      return false;
    }
    s.insert(v);
  }
  return (s.size() == 13);
}

}
