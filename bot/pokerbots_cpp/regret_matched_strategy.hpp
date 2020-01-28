#pragma once

#include <unordered_map>
#include <string>
#include <functional>

#include "infoset.hpp"

namespace pb {
namespace cfr {

typedef std::function<std::string(const EvInfoSet&)> BucketFunction;

class RegretMatchedStrategy {
 public:
  RegretMatchedStrategy(const BucketFunction& bucket_function) : bucket_function_(bucket_function) {}
  RegretMatchedStrategy() = default;

  int Size() const { return regrets_.size(); }

  // Pass in an infoset.
  void AddRegret(const EvInfoSet& infoset, const ActionRegrets& r);
  ActionRegrets GetStrategy(const EvInfoSet& infoset);

  // Pass in a bucket string.
  void AddRegret(const std::string& bucket, const ActionRegrets& r);
  ActionRegrets GetStrategy(const std::string& bucket);

  void Save(const std::string& filename);
  void Load(const std::string& filename);

  bool HasBucket(const std::string& bucket) const { return regrets_.count(bucket) != 0; }

 private:
  std::unordered_map<std::string, ActionRegrets> regrets_;
  BucketFunction bucket_function_ = BucketMedium;
};


class RegretMatchedStrategyKmeans : public RegretMatchedStrategy {
 public:
  RegretMatchedStrategyKmeans() : RegretMatchedStrategy(BucketMedium) {
    centroids_ = LoadOpponentCentroids();
    buckets_ = LoadOpponentBuckets();
  }

  RegretMatchedStrategyKmeans(const BucketFunction&) : RegretMatchedStrategy(BucketMedium) {}

  void AddRegret(const EvInfoSet& infoset, const ActionRegrets& r);
  ActionRegrets GetStrategy(const EvInfoSet& infoset);

 private:
  Centroids centroids_;
  OpponentBuckets buckets_;
};

}
}
