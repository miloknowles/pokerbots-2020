#include "hand_clustering.hpp"


int main(int argc, char const *argv[]) {
  const auto& key_to_bucket = pb::cfr::LoadOpponentBuckets();
  pb::cfr::GenerateSamples(50000, key_to_bucket);

  const auto& samples = pb::cfr::ReadSamples();
  const auto& result = pb::cfr::kmeans(samples, 1000, 10);

  const pb::cfr::Centroids& centroids = result.first;
  const pb::cfr::Clusters& clusters = result.second;

  for (int i = 0; i < clusters.size(); ++i) {
    printf("Cluster %d size = %zu\n", i, clusters.at(i).size());
    pb::cfr::Print(centroids.at(i));
  }

  pb::cfr::WriteCentroids(centroids);

  return 0;
}
