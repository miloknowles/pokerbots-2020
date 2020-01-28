#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "cfr.hpp"
#include "pbots_calc.h"

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


bool IsPossible(const std::string& hand, const std::string& board, const std::string& key) {
  std::unordered_set<std::string> dead;
  dead.emplace(hand.substr(0, 2));
  dead.emplace(hand.substr(2, 2));
  for (int i = 0; i < (board.size() / 2); ++i) {
    dead.emplace(board.substr(2*i, 2));
  }
  // std::cout << "Dead:" << std::endl;
  // for (const std::string& s : dead) {
  //   std::cout << s << " ";
  // }
  // std::cout << std::endl;

  // CASE 1: Pair of cards ==> need at least two of the remaining.
  if (key.size() == 2) {
    int remaining = 0;
    for (const std::string& suit : {"c", "h", "d", "s"}) {
      if (dead.count(key[0] + suit) == 0) { ++remaining; }
    }
    return remaining >= 2;
  } else {
    // CASE 2: Suited combination of cards ==> if at least one suited combination exists.
    if (key[2] == 's') {
      for (const std::string& suit : {"c", "h", "d", "s"}) {
        const std::string c1 = key.substr(0, 1) + suit;
        const std::string c2 = key.substr(1, 1) + suit;
        if ((dead.count(c1) == 0) && (dead.count(c2) == 0)) {
          return true;
        }
      }
    // CASE 3: Unsuited combination of cards ==> if at least one unsuited combination exists.
    } else {
      for (const std::string& suit1 : {"c", "h", "d", "s"}) {
        for (const std::string& suit2 : {"c", "h", "d", "s"}) {
          const std::string c1 = key.substr(0, 1) + suit1;
          const std::string c2 = key.substr(1, 1) + suit2;
          if ((suit1 != suit2) && ((dead.count(c1) == 0) && (dead.count(c2) == 0))) {
            return true;
          }
        }
      }
    }
  }

  return false;
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
    if (!IsPossible(hand, board, key)) {
      continue;
    }
    const float ev = PbotsCalcEquity(hand + ":" + key, board, "", 100);
    strength.at(id - 1) += ev;
    totals.at(id - 1) += 1;
  }

  for (int i = 0; i < strength.size(); ++i) {
    strength[i] /= totals[i];
  }

  return strength;
}


void GenerateSamples(int N, const OpponentBuckets& buckets) {
  std::vector<StrengthVector> samples;

  for (int n = 0; n < N; ++n) {
    if (n % 100 == 0) {
      printf("Finished %d/%d\n", n, N);
    }

    const RoundState& state = CreateNewRound(0);
    
    for (const int street : {0, 3, 4, 5}) {
      std::string board = "";
      for (int i = 0; i < street; ++i) { board += state.deck[i]; }
      const std::string& h1 = state.hands[0][0] + state.hands[0][1];
      const std::string& h2 = state.hands[1][0] + state.hands[1][1];
      const StrengthVector& strength1 = ComputeStrengthVector(buckets, h1, board);
      const StrengthVector& strength2 = ComputeStrengthVector(buckets, h2, board);

      samples.emplace_back(strength1);
      samples.emplace_back(strength2);
    }
  }

  printf("Generated %zu samples\n", samples.size());

  std::ofstream out("./strength_vector_samples.txt");
  for (const StrengthVector& v : samples) {
    for (int i = 0; i < v.size(); ++i) {
      out << v[i];
      if (i < (v.size() - 1)) {
        out << " ";
      }
    }
    out << std::endl;
  }
  out.close();
  std::cout << "Write samples to disk" << std::endl;
}


void Print(const StrengthVector& strength) {
  for (const float s : strength) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
}


}
}

int main(int argc, char const *argv[]) {
  const auto& key_to_bucket = pb::cfr::LoadOpponentBuckets();
  const pb::cfr::StrengthVector strength = pb::cfr::ComputeStrengthVector(key_to_bucket, "QcQd", "");
  pb::cfr::GenerateSamples(10000, key_to_bucket);

  return 0;
}
