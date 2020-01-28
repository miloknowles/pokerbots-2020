#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <boost/algorithm/string.hpp>

namespace pb {
namespace cfr {

typedef std::array<float, 8> StrengthVector;
typedef std::unordered_map<std::string, int> OpponentBuckets;
typedef std::unordered_map<int, StrengthVector> Centroids;
typedef std::unordered_map<int, std::vector<int>> Clusters;

OpponentBuckets LoadOpponentBuckets();

Centroids LoadOpponentCentroids();

bool IsPossible(const std::string& hand, const std::string& board, const std::string& key);

StrengthVector ComputeStrengthVector(const OpponentBuckets& buckets, const std::string& hand, const std::string& board);

void GenerateSamples(int N, const OpponentBuckets& buckets);

float Distance(const StrengthVector& v1, const StrengthVector& v2);

std::vector<StrengthVector> ReadSamples();

std::pair<Centroids, Clusters> kmeans(const std::vector<StrengthVector>& samples,
                                      int num_iters, int num_clusters);

void WriteCentroids(const Centroids& centroids);

void Print(const StrengthVector& strength);

std::string BucketHandKmeans(const Centroids& centroids, const OpponentBuckets& buckets,
                             const std::string& hand, const std::string& board);

std::string BucketHandKmeans(const Centroids& centroids, const OpponentBuckets& buckets,
                             const StrengthVector& strength);

}
}
