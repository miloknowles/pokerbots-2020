#include <fstream>
#include <utility>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "cfr.hpp"
#include "pbots_calc.h"
#include "hand_clustering.hpp"

namespace pb {
namespace cfr {

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


// Compute euclidean distance.
float Distance(const StrengthVector& v1, const StrengthVector& v2) {
  float total = 0;
  for (int i = 0; i < v1.size(); ++i) {
    total += std::pow(v1.at(i) - v2.at(i), 2);
  }
  return std::pow(total, 0.5);
}

StrengthVector Mean(const std::vector<StrengthVector>& samples, const std::vector<int>& indices) {
  StrengthVector mean;
  std::fill(mean.begin(), mean.end(), 0);

  for (const int idx : indices) {
    const StrengthVector to_add = samples.at(idx);
    for (int i = 0; i < to_add.size(); ++i) {
      mean.at(i) += to_add.at(i);
    }
  }

  for (int i = 0; i < mean.size(); ++i) {
    mean.at(i) /= static_cast<float>(indices.size());
  }

  return mean;
}


std::vector<StrengthVector> ReadSamples() {
  std::vector<StrengthVector> out;

  std::string line;
  std::ifstream infile("./strength_vector_samples.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));
    assert(strs.size() == 8);

    StrengthVector v;
    for (int i = 0; i < 8; ++i) {
      v.at(i) = std::stof(strs.at(i));
    }

    out.emplace_back(v);
  }

  printf("Read in %zu samples\n", out.size());
  return out;
}

std::pair<Centroids, Clusters> kmeans(const std::vector<StrengthVector>& samples, int num_iters, int num_clusters) {
  std::unordered_map<int, StrengthVector> mediods;
  std::unordered_map<int, std::vector<int>> clusters;

  // Make initial centroids.
  for (int ci = 0; ci < num_clusters; ++ci) {
    const int random_idx = rand() % samples.size();
    mediods.emplace(ci, samples.at(random_idx));
  }

  std::cout << "Made initial centroids" << std::endl;

  bool something_did_move = false;

  for (int iter = 0; iter < num_iters; ++iter) {
    printf("Doing kmeans iter %d/%d\n", iter, num_iters);
    something_did_move = false;

    clusters.clear();
    for (int ci = 0; ci < num_clusters; ++ci) { clusters.emplace(ci, std::vector<int>()); }

    // Assign samples to clusters.
    for (int si = 0; si < samples.size(); ++si) {
      const StrengthVector& sv = samples.at(si);
      float min_dist = std::numeric_limits<float>::max();
      int min_dist_cluster = 0;
      for (int ci = 0; ci < mediods.size(); ++ci) {
        const float dist = Distance(mediods.at(ci), sv);
        if (dist < min_dist) {
          min_dist = dist;
          min_dist_cluster = ci;
        }
      }

      clusters.at(min_dist_cluster).emplace_back(si);
    }

    // Recompute cluster centroids.
    for (int ci = 0; ci < mediods.size(); ++ci) {
      const StrengthVector new_mean = Mean(samples, clusters.at(ci));

      // Check if cluster mean changed.
      if (Distance(new_mean, mediods.at(ci)) > 1e-5) {
        something_did_move = true;
      }
      mediods.at(ci) = new_mean;
    }

    if (!something_did_move) {
      std::cout << "Cluster means didn't move, done" << std::endl;
      break;
    }
  }

  return std::make_pair(mediods, clusters);
}


void WriteCentroids(const Centroids& centroids) {
  std::ofstream out("./cluster_centroids_" + std::to_string(centroids.size()) + ".txt");

  for (const auto& it : centroids) {
    const int id = it.first;
    const StrengthVector c = it.second;

    out << id << " ";

    for (int i = 0; i < c.size(); ++i) {
      out << c[i];
      if (i < (c.size() - 1)) {
        out << " ";
      }
    }
    out << std::endl;
  }

  out.close();
  std::cout << "Write centroids to disk" << std::endl;
}


void Print(const StrengthVector& strength) {
  for (const float s : strength) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
}


}
}
