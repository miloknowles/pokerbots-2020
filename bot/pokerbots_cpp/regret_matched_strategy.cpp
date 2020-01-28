#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <boost/algorithm/string.hpp>

#include "regret_matched_strategy.hpp"

namespace pb {
namespace cfr {

void RegretMatchedStrategy::AddRegret(const EvInfoSet& infoset, const ActionRegrets& r) {
  const std::string bucket = bucket_function_(infoset);
  AddRegret(bucket, r);
}


ActionRegrets RegretMatchedStrategy::GetStrategy(const EvInfoSet& infoset) {
  const std::string bucket = bucket_function_(infoset);
  return GetStrategy(bucket);
}

void RegretMatchedStrategy::AddRegret(const std::string& bucket, const ActionRegrets& r) {
  if (regrets_.count(bucket) == 0) {
    ActionRegrets zeros = { 0, 0, 0, 0, 0, 0 };
    regrets_.emplace(bucket, zeros);
  }

  // CFR+ regret matching.
  // https://arxiv.org/pdf/1407.5042.pdf
  ActionRegrets& rplus = regrets_.at(bucket);
  for (int i = 0; i < rplus.size(); ++i) {
    rplus.at(i) = std::fmax(0.0f, rplus.at(i) + r.at(i));
  }
}


ActionRegrets RegretMatchedStrategy::GetStrategy(const std::string& bucket) {
  if (regrets_.count(bucket) == 0) {
    ActionRegrets zeros = { 0, 0, 0, 0, 0, 0 };
    regrets_.emplace(bucket, zeros);
  }

  const ActionRegrets& total_regret = regrets_.at(bucket);

  ActionRegrets rplus;
  double denom = 0;
  for (int i = 0; i < total_regret.size(); ++i) {
    const double iplus = std::fmax(0.0f, total_regret[i]);
    denom += iplus;
    rplus[i] = iplus;
  }

  // If no actions w/ positive regret, return uniform.
  if (denom <= 1e-3) {
    std::fill(rplus.begin(), rplus.end(), 1.0f / static_cast<float>(rplus.size()));
    return rplus;
  } else {
    // Normalize by the total.
    for (int i = 0; i < total_regret.size(); ++i) {
      rplus[i] /= denom;
    }
    return rplus;
  }
}

void RegretMatchedStrategy::Save(const std::string& filename) {
  std::ofstream out(filename);

  for (const auto& it : regrets_) {
    const std::string key = it.first;
    const ActionRegrets& regrets = it.second;
    out << key;
    for (const double r : regrets) {
      out << " " << r;
    }
    out << std::endl;
  }

  printf("Saved regrets for %zu buckets\n", regrets_.size());
  out.close();
}


void RegretMatchedStrategy::Load(const std::string& filename) {
  std::string line;
  std::ifstream infile(filename);

  while (std::getline(infile, line)) {
    std::istringstream iss(line);

    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));

    const std::string key = strs.at(0);
    cfr::ActionRegrets regrets_this_key;
    assert(strs.size() == (1 + regrets_this_key.size()));
    for (int i = 0; i < regrets_this_key.size(); ++i) {
      regrets_this_key.at(i) = std::stod(strs.at(i + 1));
    }
    regrets_.emplace(key, regrets_this_key);
  }

  printf("Read in regrets for %zu buckets\n", regrets_.size());
}

void RegretMatchedStrategyKmeans::AddRegret(const EvInfoSet& infoset, const ActionRegrets& r) {
  // for (const auto& it : centroids_) {
  //   Print(it.second);
  // }
  // assert(infoset.hand.size() == 4);
  std::array<std::string, 19> b = BucketBetting16(infoset);
  // b[2] = BucketHandKmeans(centroids_, buckets_, infoset.hand, infoset.board);
  b[2] = BucketHandKmeans(centroids_, buckets_, infoset.strength_vector);
  const std::string bucket = BucketJoin19(b);
  RegretMatchedStrategy::AddRegret(bucket, r);
}


ActionRegrets RegretMatchedStrategyKmeans::GetStrategy(const EvInfoSet& infoset) {
  // for (const auto& it : centroids_) {
  //   Print(it.second);
  // }
  // assert(infoset.hand.size() == 4);
  std::array<std::string, 19> b = BucketBetting16(infoset);
  // b[2] = BucketHandKmeans(centroids_, buckets_, infoset.hand, infoset.board);
  b[2] = BucketHandKmeans(centroids_, buckets_, infoset.strength_vector);
  const std::string bucket = BucketJoin19(b);
  return RegretMatchedStrategy::GetStrategy(bucket);
}

}
}
