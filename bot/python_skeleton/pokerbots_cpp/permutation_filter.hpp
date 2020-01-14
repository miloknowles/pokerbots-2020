#pragma once

#include <vector>
#include <map>
#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <utility>


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

// Pass in concatenated cards like AcAd4s5h6d.
inline std::string MapToTrueStrings(const Permutation& p, const std::string& strs) {
  std::string out;
  for (int i = 0; i < strs.size(); i += 2) {
    const char mapped_rank = p.at(RANK_STR_TO_VAL.at(strs[i]));
    out += (std::string(1, mapped_rank) + strs.at(i + 1));
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
    out[1] = RANK_STR_TO_VAL[winner_hole_cards[1]];
    return out;
  }

  HandValues GetLoserValues() const {
    HandValues out;
    out[0] = RANK_STR_TO_VAL[loser_hole_cards[0]];
    out[1] = RANK_STR_TO_VAL[loser_hole_cards[1]];
    return out;
  }

  BoardValues GetBoardValues() const {
    BoardValues out;
    for (int i = 0; i < 10; i += 2) {
      out[i] = RANK_STR_TO_VAL[board_cards[i]];
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
  PermutationFilter(int N) : N_(N), particles_(N), weights_(N), nonzero_(N), gen_(rd_()) {
    // Sample the initial population of particles.
    for (int i = 0; i < N; ++i) {
      particles_.at(i) = PriorSample();
    }
  }

  // Generate a random permutation from the prior distribution.
  Permutation PriorSample() {
    std::vector<uint8_t> orig_perm = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<uint8_t> prop_perm;

    std::geometric_distribution<uint8_t> distribution(0.25);
    Permutation seed;
    for (int i = 0; i < 13; ++i) {
      seed[i] = distribution(generator_) - 1;
    }

    for (const uint8_t s : seed) {
      const size_t pop_i = static_cast<size_t>(s) % orig_perm.size();
      prop_perm.emplace_back(orig_perm.at(pop_i));
      if (orig_perm.size() > 1) {
        const auto it = std::next(orig_perm.begin(), pop_i);
        orig_perm.erase(it);
      }
    }

    assert(prop_perm.size() == 13);
    Permutation out;
    for (int i = 0; i < 13; ++i) {
      out[i] = prop_perm[i];
    }
    
    return out;
  }

  double ComputePrior(const Permutation& p) const {
    double prob = 1.0;

    std::vector<uint8_t> orig_perm = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    for (uint8_t perm_val = 0; perm_val < 13; ++perm_val) {
      const uint8_t true_val = p[perm_val];
      const auto& it = std::find(orig_perm.begin(), orig_perm.end(), true_val);
      const size_t idx = std::distance(orig_perm.begin(), it);
      if (orig_perm.size() > 1) { orig_perm.erase(it); }

      const double s = static_cast<double>(idx);
      const double p1 = 0.25 * std::pow(0.75, s);
      const double p2 = 0.25 * std::pow(0.75, s + 2 * orig_perm.size());
      const double p3 = 0.25 * std::pow(0.75, s + 3 * orig_perm.size());
      const double p4 = 0.25 * std::pow(0.75, s + 4 * orig_perm.size());
      const double p5 = 0.25 * std::pow(0.75, s + 5 * orig_perm.size());
      prob *= (p1 + p2 + p3 + p4 + p5);
    }

    return prob;
  }

  bool SatisfiesResult(const Permutation& p, const ShowdownResult& r) const {
    const std::string query = MapToTrueStrings(p, r.winner_hole_cards) + ":" + MapToTrueStrings(p, r.loser_hole_cards);
    const std::string& board = MapToTrueStrings(p, r.board_cards);
    const bool loser_wins = PbotsCalcEquity(query, board, "", 1);
    return !loser_wins;
  }

  bool SatisfiesAll(const Permutation& p) const {
    for (const ShowdownResult& r : results_) {
      if (!SatisfiesResult(p, r)) {
        return false;
      }
    }
    return true;
  }

  int Nonzero() const { return nonzero_; }

  // For a permutation that satisfies all constraints, we don't want to swap values between the
  // hands or the hand and board. Instead, swap cards within hands or within the board.
  Permutation MakeProposalFromValid(const Permutation& p, const ShowdownResult& r) {
    const HandValues& win_hand = r.GetWinnerValues();
    const HandValues& los_hand = r.GetLoserValues();
    const BoardValues& board = r.GetBoardValues();

    // std::uniform_int_distribution<> sampler13(0, 12);
    std::uniform_int_distribution<> sampler9(0, 8);
    std::uniform_int_distribution<> sampler4(0, 3);
    std::uniform_int_distribution<> sampler5(0, 4);

    const int which = sampler9(gen_);
    int vi, vj;

    // Swap winner values.
    if (which < 2) {
      const int i = sampler4(gen_) % 2;
      vi = win_hand[i];
      vj = win_hand[(i + 1) % 2];

    // Swap loser values.
    } else if (which < 4) {
      const int i = sampler4(gen_) % 2;
      vi = los_hand[i];
      vj = los_hand[(i + 1) % 2];

    // Swap board values.
    } else {
      const int i = sampler5(gen_);
      const int j = sampler5(gen_);
      vi = board[i];
      vj = board[j];
    }

    // TODO(milo): Swap remaining 4 values if performance is worse.

    Permutation prop = p;
    const uint8_t ti = prop[vi];
    const uint8_t tj = prop[vj];
    prop[vi] = tj;
    prop[vj] = ti;

    return prop;
  }

  // Make a proposal permutation by swapping a card from the winner's hand, loser's hand, or
  // remaining deck.
  Permutation MakeProposalFromInvalid(const Permutation& p, const ShowdownResult& r) {
    const HandValues& win_hand = r.GetWinnerValues();
    const HandValues& los_hand = r.GetLoserValues();
    const BoardValues& board = r.GetBoardValues();

    std::array<bool, 13> other_mask;
    other_mask.fill(true);
    for (const uint8_t v : win_hand) { other_mask[v] = false; }
    for (const uint8_t v : los_hand) { other_mask[v] = false; }
    for (const uint8_t v : board) { other_mask[v] = false; }
    std::vector<uint8_t> other_vals;

    for (uint8_t i = 0; i < other_mask.size(); ++i) {
      if (other_mask[i]) { other_vals.emplace_back(i); }
    }

    std::uniform_int_distribution<> sampler4(0, 4);
    std::uniform_int_distribution<> sampler6(0, 6);
    const int i = sampler4(gen_) % 2;
    const int j = sampler6(gen_);

    std::uniform_real_distribution<> real(0, 1);

    uint8_t vi, vj;

    // Swap winner hand with others.
    if (real(gen_) < 0.5) {
      vi = win_hand[i];
      other_vals.insert(other_vals.end(), win_hand.begin(), win_hand.end());
      vj = other_vals[j];
    // Swap loser hand with others.
    } else {
      vi = los_hand[i];
      other_vals.insert(other_vals.end(), los_hand.begin(), los_hand.end());
      vj = other_vals[j];
    }

    Permutation prop = p;
    const uint8_t ti = prop[vi];
    const uint8_t tj = prop[vj];
    prop[vi] = tj;
    prop[vj] = ti;

    return prop;
  }

  // Do Metropolis-Hastings rejection sampling on a proposed permutation.
  std::pair<Permutation, bool> MetropolisHastings(const Permutation& orig_perm, const Permutation& prop_perm) {
    const double prior_prop = ComputePrior(prop_perm);
    const double prior_orig = ComputePrior(orig_perm);

    const double A_ij = std::min(1.0, prior_prop / prior_orig);
    std::uniform_real_distribution<> real(0, 1);
    if (real(gen_) < A_ij) {
      if (SatisfiesAll(prop_perm)) {
        return std::make_pair<Permutation, bool>(Permutation(prop_perm), true);
      }
    }

    return std::make_pair<Permutation, bool>(Permutation(orig_perm), false);
  }


 private:
  int N_;

  std::vector<Permutation> particles_;
  std::vector<double> weights_;
  int nonzero_ = 0;
  std::vector<ShowdownResult> results_ = {};

  std::vector<std::size_t> dead_indices_ = {};

  std::default_random_engine generator_{};
  std::random_device rd_{};
  std::mt19937 gen_;
};

}