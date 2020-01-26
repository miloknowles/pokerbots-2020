#include "regret_matched_strategy.hpp"

namespace pb {
namespace cfr {

void RegretMatchedStrategy::AddRegret(const EvInfoSet& infoset, const ActionRegrets& r) {
  const std::string bucket = BucketSmallJoin(BucketInfoSetSmall(infoset));
  if (regrets_.count(bucket) == 0) {
    ActionRegrets zeros = { 0, 0, 0, 0, 0, 0 };
    regrets_.emplace(bucket, zeros);
  }

  // CFR+ regret matching.
  // https://arxiv.org/pdf/1407.5042.pdf
  ActionRegrets rplus = regrets_.at(bucket);
  for (int i = 0; i < rplus.size(); ++i) {
    rplus.at(i) = std::fmax(0.0f, rplus.at(i) + r.at(i));
  }
}


ActionRegrets RegretMatchedStrategy::GetStrategy(const EvInfoSet& infoset) {
  const std::string bucket = BucketSmallJoin(BucketInfoSetSmall(infoset));

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

}
}
