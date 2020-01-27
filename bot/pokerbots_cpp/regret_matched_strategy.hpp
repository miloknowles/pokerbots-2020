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

  void AddRegret(const EvInfoSet& infoset, const ActionRegrets& r);

  ActionRegrets GetStrategy(const EvInfoSet& infoset);

  void Save(const std::string& filename);
  void Load(const std::string& filename);

  bool HasBucket(const std::string& bucket) const { return regrets_.count(bucket) != 0; }

 private:
  std::unordered_map<std::string, ActionRegrets> regrets_;
  BucketFunction bucket_function_ = BucketMedium;
};

}
}
