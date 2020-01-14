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


Permutation PermutationFilter::PriorSample() {
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


double PermutationFilter::ComputePrior(const Permutation& p) const {
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


Permutation PermutationFilter::MakeProposalFromValid(const Permutation& p, const ShowdownResult& r) {
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


Permutation PermutationFilter::MakeProposalFromInvalid(const Permutation& p, const ShowdownResult& r) {
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


std::pair<Permutation, bool> PermutationFilter::MetropolisHastings(const Permutation& orig_perm, const Permutation& prop_perm) {
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

}
