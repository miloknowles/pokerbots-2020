#include "./permutation_filter.hpp"


int main(int argc, char const *argv[]) {
  int N = 1000000;

  for (int i = 0; i < N; ++i) {
    const std::string query = "AcTd:QsQd";
    const std::string board = "2h3s8c9dTh";
    const std::string dead = "";
    const float ev = pb::PbotsCalcEquity(query, board, dead, 1);
    // std::cout << ev << std::endl;
  }
}
