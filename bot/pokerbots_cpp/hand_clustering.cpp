#include <string>
#include <unordered_map>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "cfr.hpp"

namespace pb {
namespace cfr {

typedef std::array<float, 8> StrengthVector;
typedef std::unordered_map<std::string, int> OpponentBuckets;

// Load in a map that converts 169 hands to one of 8 clusters.
OpponentBuckets LoadOpponentBuckets() {
  OpponentBuckets out;

  std::string line;
  std::ifstream infile("opponent_clusters.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));
    assert(strs.size() == 2);

    const std::string& key = strs.at(0);
    const int id = std::stoi(strs.at(1));
    out.emplace(key, id);
  }

  assert(out.size() == 169);

  return out;
}


StrengthVector ComputeStrengthVector(const OpponentBuckets& buckets, const std::string& hand, const std::string& board) {
  StrengthVector strength;
  StrengthVector totals;
  std::fill(strength.begin(), strength.end(), 0);

  for (const auto& it : buckets) {
    std::string key = it.first;
    const int id = it.second;

    if (key[0] == key[1] && key[2] == 'o') {
      key = key.substr(0, 2);
    }
    
    const std::string query = hand + ":" + key;
    // std::cout << query << std::endl;
    const float ev =  PbotsCalcEquity(hand + ":" + key, board, "", 100);
    strength.at(id - 1) += ev;
    totals.at(id - 1) += 1;
  }

  for (int i = 0; i < strength.size(); ++i) {
    strength[i] /= totals[i];
  }

  return strength;
}


}
}


int main(int argc, char const *argv[]) {
  const auto& key_to_bucket = pb::cfr::LoadOpponentBuckets();
  const pb::cfr::StrengthVector strength = pb::cfr::ComputeStrengthVector(key_to_bucket, "QcQd", "");

  for (const float s : strength) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
  return 0;
}
