#include "./permutation_filter.hpp"


int main(int argc, char const *argv[]) {
  int N = 10000;

  for (int i = 0; i < N; ++i) {
    const std::string query = "AcTd:QsQd";
    const std::string board = "2h3s8c9dTh";
    const std::string dead = "";
    const float ev = pb::PbotsCalcEquity(query, board, dead, 1);
    // std::cout << ev << std::endl;
  }

  pb::PermutationFilter pf(5000);

  const pb::Permutation& perm = pf.PriorSample();
  const double prior = pf.ComputePrior(perm);
  std::cout << "Prior: " << prior << std::endl;
}
