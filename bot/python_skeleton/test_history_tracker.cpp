#include "./pokerbots_cpp_python/gtest/gtest.h"

#include "history_tracker.hpp"

using namespace pb;

TEST(HistoryTrackerTest, testUpdateSb) {
  // Small blind: we will only update the tracker on our actions.
  HistoryTracker tracker_(false);

  // Initialization - small and big blinds paid out.
  const auto& v0 = tracker_.Vector();
  EXPECT_EQ(1, v0[0]);
  EXPECT_EQ(2, v0[1]);

  // Make the SB call, BB bet.
  tracker_.Update(2, 6, 0);
  const auto& v1 = tracker_.Vector();
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(1, v1[2]);
  EXPECT_EQ(4, v1[3]);
  tracker_.Print();

  // Make the SB raise, BB call.
  tracker_.Update(10, 10, 0);
  const auto& v2 = tracker_.Vector();
  EXPECT_EQ(1, v2[0]);
  EXPECT_EQ(2, v2[1]);
  EXPECT_EQ(1, v2[2]);
  EXPECT_EQ(4, v2[3]);
  EXPECT_EQ(8, v2[4]);
  EXPECT_EQ(4, v2[5]);
  tracker_.Print();

  // BB checks the flop.
  tracker_.Update(10, 10, 3);
  tracker_.Print();

  // SB bets 7, BB raises to 13.
  tracker_.Update(17, 23, 3);
  const auto& v3 = tracker_.Vector();
  EXPECT_EQ(0, v3[8]);
  EXPECT_EQ(7, v3[9]);
  EXPECT_EQ(13, v3[10]);
  tracker_.Print();

  // SB calls, done with flop, go to turn, BB bets 20.
  tracker_.Update(23, 43, 4);
  const auto& v4 = tracker_.Vector();
  EXPECT_EQ(6, v4[11]);
  EXPECT_EQ(0, v4[12]);
  EXPECT_EQ(20, v4[16]);
  tracker_.Print();
}


TEST(HistoryTrackerTest, testUpdateBb) {
  HistoryTracker tracker_(true);

  // Initialization - small and big blinds paid out.
  const auto& v0 = tracker_.Vector();
  EXPECT_EQ(1, v0[0]);
  EXPECT_EQ(2, v0[1]);

  // SB called.
  tracker_.Update(2, 2, 0);
  const auto& v1 = tracker_.Vector();
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(1, v1[2]);
  tracker_.Print();

  // BB bet and SB raised.
  tracker_.Update(10, 20, 0);
  const auto& v2 = tracker_.Vector();
  EXPECT_EQ(1, v2[0]);
  EXPECT_EQ(2, v2[1]);
  EXPECT_EQ(1, v2[2]);
  EXPECT_EQ(8, v2[3]);
  EXPECT_EQ(18, v2[4]);
  tracker_.Print();

  // BB called on preflop and then BB checks the flop.
  tracker_.Update(20, 20, 3);
  const auto& v3 = tracker_.Vector();
  EXPECT_EQ(10, v3[5]);
  EXPECT_EQ(0, v3[8]);
  tracker_.Print();
}
