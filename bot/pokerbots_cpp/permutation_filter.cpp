#include <fstream>

#include "./permutation_filter.hpp"
#include "pbots_calc.h"

namespace pb {


static void PrintResult(const ShowdownResult& r) {
  std::cout << "Winner hand: " << r.winner_hole_cards << " Loser hand: " << r.loser_hole_cards << " Board: " << r.board_cards << std::endl;
}


static void PrintValues(const HandValues& w, const HandValues& l, const BoardValues& b) {
  for (const uint8_t wv : w) {
    std::cout << static_cast<int>(wv) << " ";
  }
  for (const uint8_t lv : l) {
    std::cout << static_cast<int>(lv) << " ";
  }
  for (const uint8_t bv : b) {
    std::cout << static_cast<int>(bv) << " ";
  }
  std::cout << "\n";
}


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


PermutationFilter::PermutationFilter(int N) : N_(N), particles_(N), weights_(N), gen_(rd_()) {
  // Sample the initial population of particles.
  for (int i = 0; i < N; ++i) {
    particles_.at(i) = PriorSample();
    weights_.at(i) = 1.0;
    MaybeAddUnique(particles_.at(i));
  }
  // Precompute pow for some speedups.
  for (int i = 0; i < pow_precompute_.size(); ++i) {
    pow_precompute_.at(i) = std::pow(0.75, i);
  }
  // Load in the preflop equities.
  std::string hand;
  float ev;
  // NOTE: this must be copied to same location as the executable.
  std::ifstream in("./preflop_equity.txt");
  while (in >> hand >> ev) {
    preflop_ev_.emplace(hand, ev);
  }
  assert(preflop_ev_.size() == 2652);
}


Permutation PermutationFilter::PriorSample() {
  std::vector<uint8_t> orig_perm = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  std::vector<uint8_t> prop_perm;

  std::geometric_distribution<uint8_t> distribution(0.25);
  Permutation seed;
  for (int i = 0; i < 13; ++i) {
    seed[i] = distribution(generator_);
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


double PermutationFilter::ComputePrior(const Permutation& p) {
  Timer timer;
  double prob = 1.0;

  Permutation queue_pos = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  for (uint8_t perm_val = 0; perm_val < 13; ++perm_val) {
    const uint8_t true_val = p[perm_val];

    // Everything in the queue after the popped item gets moved forward.
    for (uint8_t i = true_val + 1; i < 13; ++i) {
      queue_pos[i] -= 1;
    }

    const int s = static_cast<int>(queue_pos[true_val]);
    const int queue_size = static_cast<int>(13 - perm_val);
    double prob_sum = 0;
    for (int i = 0; i < 5; ++i) {
      const int s_wrap = s + (i * queue_size);
      if (s_wrap < pow_precompute_.size()) {
        prob_sum += 0.25 * pow_precompute_[s_wrap];
      }
    }
    prob *= prob_sum;
  }

  UpdateProfile("ComputePrior", timer.Elapsed());
  return prob;
}

bool PermutationFilter::SatisfiesResult(const Permutation& p, const ShowdownResult& r) {
  Timer timer;

  const std::string& query = MapToTrueStrings(p, r.winner_hole_cards) + ":" + MapToTrueStrings(p, r.loser_hole_cards);
  const std::string& board = MapToTrueStrings(p, r.board_cards);
  const float ev = PbotsCalcEquity(query, board, "", 1);

  UpdateProfile("SatisfiesResult", timer.Elapsed());
  return ev > 0;
}

bool PermutationFilter::SatisfiesAll(const Permutation& p) {
  Timer timer;
  for (const ShowdownResult& r : results_) {
    if (!SatisfiesResultOmp(p, r)) { return false; }
  }
  UpdateProfile("SatisfiesAll", timer.Elapsed());
  return true;
}


Permutation PermutationFilter::MakeProposalFromValid(const Permutation& p, const ShowdownResult& r) {
  Timer timer;

  const HandValues win_hand = r.GetWinnerValues();
  const HandValues los_hand = r.GetLoserValues();
  const BoardValues board = r.GetBoardValues();

  std::uniform_int_distribution<> sampler13(0, 12);
  // std::uniform_int_distribution<> sampler9(0, 8);
  std::uniform_int_distribution<> sampler4(0, 3);
  std::uniform_int_distribution<> sampler5(0, 4);

  const int which = sampler13(gen_);
  uint8_t vi, vj;

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
  } else if (which < 9) {
    const int i = sampler5(gen_);
    const int j = sampler5(gen_);
    vi = board[i];
    vj = board[j];
  } else {
    std::array<bool, 13> other_mask;
    other_mask.fill(true);
    for (const uint8_t v : win_hand) { other_mask[v] = false; }
    for (const uint8_t v : los_hand) { other_mask[v] = false; }
    for (const uint8_t v : board) { other_mask[v] = false; }
    std::vector<uint8_t> other_vals;
    for (uint8_t i = 0; i < other_mask.size(); ++i) {
      if (other_mask[i]) { other_vals.emplace_back(i); }
    }
    const int i = sampler4(gen_);
    const int j = sampler4(gen_);
    vi = other_vals[i];
    vj = other_vals[j];
  }

  Permutation prop = p;
  const uint8_t ti = prop[vi];
  const uint8_t tj = prop[vj];
  prop[vi] = tj;
  prop[vj] = ti;

  UpdateProfile("MakeProposalFromValid", timer.Elapsed());
  return prop;
}


Permutation PermutationFilter::MakeProposalFromInvalid(const Permutation& p, const ShowdownResult& r) {
  Timer timer;

  const HandValues win_hand = r.GetWinnerValues();
  const HandValues los_hand = r.GetLoserValues();
  const BoardValues board = r.GetBoardValues();

  std::array<bool, 13> other_mask;
  other_mask.fill(true);
  for (const uint8_t v : win_hand) { other_mask[v] = false; }
  for (const uint8_t v : los_hand) { other_mask[v] = false; }
  for (const uint8_t v : board) { other_mask[v] = false; }
  std::vector<uint8_t> other_vals;
  for (uint8_t i = 0; i < other_mask.size(); ++i) {
    if (other_mask[i]) { other_vals.emplace_back(i); }
  }

  std::uniform_int_distribution<> sampler4(0, 3);
  std::uniform_int_distribution<> sampler6(0, 5);
  const int i = sampler4(gen_) % 2;
  const int j = sampler6(gen_);

  std::uniform_real_distribution<> real(0, 1);

  uint8_t vi, vj;

  // Swap winner hand with others.
  if (real(gen_) < 0.5) {
    vi = win_hand.at(i);
    other_vals.insert(other_vals.end(), win_hand.begin(), win_hand.end());
    vj = other_vals.at(j);
  // Swap loser hand with others.
  } else {
    vi = los_hand.at(i);
    other_vals.insert(other_vals.end(), los_hand.begin(), los_hand.end());
    vj = other_vals.at(j);
  }

  Permutation prop(p);
  const uint8_t ti = prop.at(vi);
  const uint8_t tj = prop.at(vj);
  prop.at(vi) = tj;
  prop.at(vj) = ti;

  UpdateProfile("MakeProposalFromInvalid", timer.Elapsed());
  return prop;
}


std::pair<Permutation, bool> PermutationFilter::MetropolisHastings(const Permutation& orig_perm, const Permutation& prop_perm) {
  Timer timer;
  const double prior_prop = ComputePrior(prop_perm);
  const double prior_orig = ComputePrior(orig_perm);

  const double A_ij = std::min(1.0, prior_prop / prior_orig);
  std::uniform_real_distribution<> real(0, 1);
  if (real(gen_) < A_ij) {
    if (SatisfiesAll(prop_perm)) {
      return std::make_pair<Permutation, bool>(Permutation(prop_perm), true);
    }
  }

  UpdateProfile("MetropolisHastings", timer.Elapsed());
  return std::make_pair<Permutation, bool>(Permutation(orig_perm), false);
}


std::pair<Permutation, bool> PermutationFilter::SampleMCMCInvalid(const Permutation& orig_perm, const ShowdownResult& r) {
  const Permutation& prop_perm = MakeProposalFromInvalid(orig_perm, r);
  return MetropolisHastings(orig_perm, prop_perm);
}


std::pair<Permutation, bool> PermutationFilter::SampleMCMCValid(const Permutation& orig_perm, const ShowdownResult& r) {
  const Permutation& prop_perm = MakeProposalFromValid(orig_perm, r);
  return MetropolisHastings(orig_perm, prop_perm);
}


void PermutationFilter::Update(const ShowdownResult& r) {
  const int nonzero = Nonzero();

  if (nonzero == 0) {
    return;
  }

  // const int num_invalid_retries = std::min(5, static_cast<int>(5 * N_ / nonzero));
  const int num_invalid_retries = 5;
  const int num_valid_retries = 2;

  for (int i = 0; i < particles_.size(); ++i) {
    // Skip particles that have zero weight (dead).
    if (weights_.at(i) <= 0) {
      continue;
    }
  
    const Permutation& p = particles_.at(i);

    // If this result will kill particle, try to fix it a few times before giving up.
    if (!SatisfiesResultOmp(p, r)) {
      MaybeRemoveUnique(p);
      
      bool did_save = false;
      for (int rt = 0; rt < num_invalid_retries; ++rt) {
        const auto& mcmc_sample = SampleMCMCInvalid(p, r);
        did_save = mcmc_sample.second;

        // Successful fix!
        if (did_save) {
          weights_.at(i) = 1;
          particles_.at(i) = mcmc_sample.first;
          MaybeAddUnique(mcmc_sample.first);
          break;
        }
      }

      if (!did_save) {
        weights_.at(i) = 0;
        dead_indices_.emplace_back(i);
      }
    
    // Result doesn't kill particle, use it to make some more samples.
    } else {
      for (int rt = 0; rt < num_valid_retries; ++rt) {
        if (dead_indices_.size() == 0) {
          break;
        }

        const auto& mcmc_sample = SampleMCMCValid(p, r);
        if (mcmc_sample.second) {
          const int dead_idx_to_replace = dead_indices_.back();
          dead_indices_.pop_back();

          weights_.at(dead_idx_to_replace) = 1;
          particles_.at(dead_idx_to_replace) = mcmc_sample.first;
          MaybeAddUnique(mcmc_sample.first);
          break;
        }
      }
    }
  }

  results_.emplace_back(r);
}

void PermutationFilter::Profile() const {
  std::cout << "\n*** PROFILING RESULTS ***" << std::endl;
  for (const auto& it : time_) {
    const std::string fn = it.first;
    const double total_t = it.second;
    const int ct = counts_.at(fn);
    const double avg_t = total_t / static_cast<double>(ct);
    std::cout << fn << std::endl;
    printf(" | TOTAL=%lf | COUNT=%d | AVG=%lf\n", total_t, ct, avg_t);
  }
}

void PermutationFilter::UpdateProfile(const std::string& fn, const double elapsed) {
  if (counts_.count(fn) == 0) {
    counts_.emplace(fn, 0);
    time_.emplace(fn, 0);
  }
  ++counts_.at(fn);
  time_.at(fn) += elapsed;
}

bool PermutationFilter::SatisfiesResultOmp(const Permutation&p, const ShowdownResult& r) {
  omp::Hand win = omp::Hand::empty();
  const std::string win_and_board = r.winner_hole_cards + r.board_cards;
  assert(win_and_board.size() == 14);
  for (int i = 0; i < 7; ++i) {
    const uint8_t code = 4 * p[RANK_STR_TO_VAL[win_and_board[2*i]]] + SUIT_STR_TO_VAL[win_and_board[2*i+1]];
    win += omp::Hand(code);
  }
  assert(win.count() == 7);
  const uint16_t win_score = omp_.evaluate(win);

  omp::Hand lose = omp::Hand::empty();
  const std::string lose_and_board = r.loser_hole_cards + r.board_cards;
  assert(lose_and_board.size() == 14);
  for (int i = 0; i < 7; ++i) {
    const uint8_t code = 4 * p[RANK_STR_TO_VAL[lose_and_board[2*i]]] + SUIT_STR_TO_VAL[lose_and_board[2*i+1]];
    lose += omp::Hand(code);
  }
  assert(lose.count() == 7);
  const uint16_t lose_score = omp_.evaluate(lose);

  return win_score >= lose_score;
}

float PermutationFilter::ComputeEvRandom(const std::string& hand,
                                        const std::string& board,
                                        const std::string& dead,
                                        const int nsamples,
                                        const int iters) {
  Timer timer;
  float ev = 0;

  // Get indices of nonzero particles.
  std::vector<int> valid_idx;
  for (int i = 0; i < weights_.size(); ++i) {
    if (weights_[i] > 0) { valid_idx.emplace_back(i); }
  }
  if (valid_idx.size() < nsamples) {
    std::cout << "WARNING: not enough valid particles to sample" << std::endl;
    return -1.0f;
  }

  std::uniform_int_distribution<> sampler(0, valid_idx.size()-1);

  for (int si = 0; si < nsamples; ++si) {
    const int unif_int = sampler(gen_);
    const int rand_idx = valid_idx.at(unif_int);
    const Permutation& perm = particles_.at(rand_idx);
    const std::string& board_m = MapToTrueStrings(perm, board);

    // For preflop, use lookup table.
    if (board_m.size() == 0) {
      ev += preflop_ev_.at(MapToTrueStrings(perm, hand));
    } else {
      const std::string& query_m = MapToTrueStrings(perm, hand) + ":xx";
      const std::string& dead_m = MapToTrueStrings(perm, dead);
      ev += PbotsCalcEquity(query_m, board_m, dead_m, iters);
    }
  }

  UpdateProfile("ComputeEvRandom", timer.Elapsed());
  return (ev / static_cast<float>(nsamples));
}

static cfr::StrengthVector Add(const cfr::StrengthVector& v1, const cfr::StrengthVector& v2) {
  cfr::StrengthVector out = v1;
  for (int i = 0; i < v2.size(); ++i) {
    out[i] += v2[i];
  }
  return out;
}

static cfr::StrengthVector Divide(const cfr::StrengthVector& v, float c) {
  cfr::StrengthVector out = v;
  for (int i = 0; i < v.size(); ++i) {
    out[i] /= c;
  }
  return out;
}

cfr::StrengthVector PermutationFilter::ComputeStrengthRandom(const std::string& hand,
                                                              const std::string& board,
                                                              const std::string& dead,
                                                              const int nsamples,
                                                              const cfr::OpponentBuckets& buckets) {
  Timer timer;
  cfr::StrengthVector strength;

  // Get indices of nonzero particles.
  std::vector<int> valid_idx;
  for (int i = 0; i < weights_.size(); ++i) {
    if (weights_[i] > 0) { valid_idx.emplace_back(i); }
  }
  if (valid_idx.size() < nsamples) {
    std::cout << "WARNING: not enough valid particles to sample" << std::endl;
    cfr::StrengthVector out;
    std::fill(out.begin(), out.end(), 0);
    return out;
  }

  std::uniform_int_distribution<> sampler(0, valid_idx.size()-1);

  for (int si = 0; si < nsamples; ++si) {
    const int unif_int = sampler(gen_);
    const int rand_idx = valid_idx.at(unif_int);
    const Permutation& perm = particles_.at(rand_idx);
    const std::string& board_m = MapToTrueStrings(perm, board);
    const std::string& hand_m = MapToTrueStrings(perm, hand);
    const std::string& dead_m = MapToTrueStrings(perm, dead);

    const cfr::StrengthVector& sampled_strength = cfr::ComputeStrengthVector(buckets, hand_m, board_m);
    strength = Add(sampled_strength, strength);
  }

  UpdateProfile("ComputeStrengthRandom", timer.Elapsed());
  return Divide(strength, static_cast<float>(nsamples));
}

} // namespace pb
