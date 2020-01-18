#pragma once

#include <vector>
#include <map>
#include <unordered_map>
// #include <unordered_set>
#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <utility>
#include <set>
#include <chrono>

#include <omp/HandEvaluator.h>
// #include <omp/EquityCalculator.h>

namespace pb {

typedef std::array<uint8_t, 13> Permutation;
typedef std::array<uint8_t, 2> HandValues;
typedef std::array<uint8_t, 5> BoardValues;


static std::map<char, uint8_t> RANK_STR_TO_VAL = {
  {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3}, {'6', 4}, {'7', 5}, {'8', 6},
  {'9', 7}, {'T', 8}, {'J', 9}, {'Q', 10}, {'K', 11}, {'A', 12}
};

static std::map<uint8_t, char> RANK_VAL_TO_STR = {
  {0, '2'}, {1, '3'}, {2, '4'}, {3, '5'}, {4, '6'}, {5, '7'}, {6, '8'},
  {7, '9'}, {8, 'T'}, {9, 'J'}, {10, 'Q'}, {11, 'K'}, {12, 'A'}
};

// Defined by OMPEval.
static std::map<char, uint8_t> SUIT_STR_TO_VAL = {
  {'s', 0}, {'h', 1}, {'c', 2}, {'d', 3}
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


// inline float OmpCalcEquity(const std::string& hand_str,
//                     const std::string& board_str,
//                     const std::string& dead_str) {
//   omp::EquityCalculator eq;
//   std::vector<omp::CardRange> ranges = { hand_str, "random" };
//   uint64_t board = omp::CardRange::getCardMask(board_str);
//   uint64_t dead = omp::CardRange::getCardMask(dead_str);
//   double stdErrMargin = 0.1;
//   double updateInterval = 1.0;
//   unsigned threads = 1;
//   eq.start(ranges, board, dead, false, stdErrMargin, nullptr, updateInterval, threads);
//   eq.wait();
//   auto r = eq.getResults();
//   return r.equity[0];
// }


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


// TODO: make more concise.
inline std::vector<uint8_t> MapToTrueValues(const Permutation& p, const HandValues& values) {
  std::vector<uint8_t> values_mapped(values.size());
  for (int i = 0; i < values.size(); ++i) {
    const uint8_t mapped = p.at(values.at(i));
    assert(mapped >= 0 && mapped <= 12);
    values_mapped.at(i) = mapped;
  }
  return values_mapped;
}


// TODO: make more concise.
inline std::vector<uint8_t> MapToTrueValues(const Permutation& p, const BoardValues& values) {
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

// Useful for profiling.
class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  void Reset() { start_ = std::chrono::steady_clock::now(); }
  void Stop() { end_ = std::chrono::steady_clock::now(); }

  double Elapsed() {
    end_ = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
};


class PermutationFilter {
 public:
  PermutationFilter(int N);

  PermutationFilter(const PermutationFilter&) = delete;

  // Generate a random permutation from the prior distribution.
  Permutation PriorSample();

  // Compute the prior probability of sampling Permutation p.
  double ComputePrior(const Permutation& p);

  // Does this permutations p satisfy the showdown result r?
  bool SatisfiesResult(const Permutation& p, const ShowdownResult& r);

  // Does permutation p satisfy ALL results seen so far?
  bool SatisfiesAll(const Permutation& p);

  int Nonzero() const { return (N_ - dead_indices_.size()); }

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
    for (const Permutation& p : particles_) { if (Equals(query, p)) { return true; } }
    return false;
  }

  void Profile() const;
  void UpdateProfile(const std::string& fn, const double elapsed);

  // Faster result checking.
  bool SatisfiesResultOmp(const Permutation&p, const ShowdownResult& r);

  // Compute expected value of a hand vs. a randomly drawn opponent across all permutation samples.
  float ComputeEvRandom(const std::string& hand,
                        const std::string& board,
                        const std::string& dead,
                        const int nsamples,
                        const int iters);

  void MaybeAddUnique(const Permutation& p) {
    std::string hash = "";
    for (int i = 0; i < p.size(); ++i) {
      hash += std::to_string(p[i]);
    }
    // std::cout << hash << std::endl;
    if (unique_.count(hash) == 0) {
      unique_.emplace(hash, 0);
    }
    ++unique_.at(hash);
  }

  void MaybeRemoveUnique(const Permutation& p) {
    std::string hash = "";
    for (int i = 0; i < p.size(); ++i) {
      hash += std::to_string(p[i]);
    }
    if (unique_.count(hash) > 0) {
      unique_.at(hash) -= 1;
      if (unique_[hash] <= 0) {
        unique_.erase(hash);
      }
    }
  }

  int Unique() const { return unique_.size(); }

 private:
  int N_;

  std::vector<Permutation> particles_;
  std::unordered_map<std::string, int> unique_;
  std::vector<double> weights_;
  std::vector<ShowdownResult> results_ = {};

  std::vector<int> dead_indices_ = {};

  std::default_random_engine generator_{};
  std::random_device rd_{};
  std::mt19937 gen_;

  std::array<double, 40> pow_precompute_;
  omp::HandEvaluator omp_;
  std::unordered_map<std::string, float> preflop_ev_{};

  std::unordered_map<std::string, double> time_;
  std::unordered_map<std::string, int> counts_;
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


inline void PrintVector(const std::vector<int>& vec) {
  for (const int t : vec) {
    std::cout << t << " ";
  }
  std::cout << "\n";
}


inline void PrintHistory(const std::vector<int>& vec) {
  const int actions_per = vec.size() / 4;

  for (int st = 0; st < 4; ++st) {
    for (int i = 0; i < actions_per; ++i) {
      std::cout << vec.at(st*actions_per + i) << " ";
    }
    std::cout << "| ";
  }
  std::cout << std::endl;
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
