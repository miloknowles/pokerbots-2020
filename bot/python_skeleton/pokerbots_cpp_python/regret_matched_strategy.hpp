#pragma once

#include <unordered_map>
#include <string>
#include <algorithm>

#include "infoset.hpp"

namespace pb {
namespace cfr {

class RegretMatchedStrategy {
 public:
  RegretMatchedStrategy() = default;

  int Size() const { return regrets_.size(); }

  void AddRegret(const EvInfoSet& infoset, const ActionRegrets& r);

  ActionRegrets GetStrategy(const EvInfoSet& infoset);

  void Save(const std::string& filename);
  void Load(const std::string& filename);

 private:
  std::unordered_map<std::string, ActionRegrets> regrets_;
};

}
}
