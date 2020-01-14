#include "./gtest/gtest.h"
#include "./permutation_filter.hpp"

using namespace pb;

template <typename T>
static bool ASSERT_VECTOR_EQ(const std::vector<T>& v1, const std::vector<T>& v2) {
  if (v1.size() != v2.size()) {
    return false;
  }
  for (int i = 0; i < v1.size(); ++i) {
    if (v1.at(i) != v2.at(i)) { return false; }
  }
  return true;
}

TEST(PermutationFilterTest, testMapToTrueValues) {
  const Permutation p = { 1, 2, 7, 0, 6, 10, 11, 12, 3, 4, 9, 8, 5 };
  const std::vector<uint8_t> values_to_map = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  const auto& mapped = MapToTrueValues(p, values_to_map);
  EXPECT_TRUE(ASSERT_VECTOR_EQ<uint8_t>(PermutationToVector(p), mapped));
}

TEST(PermutationFilterTest, testPriorSample) {
  PermutationFilter pf(100);

  for (int i = 0; i < 1000000; ++i) {
    const Permutation& p = pf.PriorSample();
    // PrintPermutation(p);
    const double prior = pf.ComputePrior(p);
    // std::cout << prior << std::endl;
    EXPECT_TRUE(PermutationIsValid(p));
  }
}

TEST(PermutationFilterTest, testMapToTrueStrings) {
  const Permutation p = { 1, 2, 7, 0, 6, 10, 11, 12, 3, 4, 9, 8, 5 };
  const std::string test_rank = "2c3c4c5c6c7c8c9cTcJcQcKcAc";
  const std::string exp_rank =  "3c4c9c2c8cQcKcAc5c6cJcTc7c";

  const std::string true_rank = MapToTrueStrings(p, test_rank);
  for (int i = 0; i < 13; ++i) {
    if (i % 2 == 1) {
      EXPECT_TRUE(true_rank.at(i) == 'c');
    } else {
      EXPECT_EQ(exp_rank.at(i), true_rank.at(i));
    }
  }
}

TEST(ShowdownResultTest, testShowdownResult) {
  const ShowdownResult r("AsKd", "2h3c", "2c3c4c5c6c");

  const HandValues& win = r.GetWinnerValues();
  const HandValues& lose = r.GetLoserValues();
  const BoardValues& board = r.GetBoardValues();

  EXPECT_EQ(win[0], 12);
  EXPECT_EQ(win[1], 11);

  EXPECT_EQ(lose[0], 0);
  EXPECT_EQ(lose[1], 1);

  EXPECT_EQ(board[0], 0);
  EXPECT_EQ(board[1], 1);
  EXPECT_EQ(board[2], 2);
  EXPECT_EQ(board[3], 3);
  EXPECT_EQ(board[4], 4);
}

TEST(PermutationFilterTest, testMakeProposalFromInvalid) {
  PermutationFilter pf(100);
  const Permutation original = pf.PriorSample();
  const ShowdownResult r("AsKd", "2h3c", "2c3c4c5c6c");

  for (int i = 0; i < 10000; ++i) {
    const Permutation p = pf.MakeProposalFromInvalid(original, r);
    const double prior = pf.ComputePrior(p);
    // std::cout << prior << std::endl;
  }
}
