#include "./gtest/gtest.h"

#include "history_tracker.hpp"

using namespace pb;

TEST(HistoryTrackerTest, testUpdateSb) {
  // Small blind: we will only update the tracker on our actions.
  HistoryTracker tracker_(false);

  // Initialization - small and big blinds paid out.
  const auto& v0 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(1, v0.size());
  EXPECT_EQ(1, v0[0][0]);
  EXPECT_EQ(2, v0[0][1]);

  tracker_.Update(1, 2, 0);
  tracker_.Print();

  // Make the SB call, BB bet.
  tracker_.Update(2, 6, 0);
  const auto& v1 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(1, v1.size());
  EXPECT_EQ(1, v1[0][0]);
  EXPECT_EQ(2, v1[0][1]);
  EXPECT_EQ(1, v1[0][2]);
  EXPECT_EQ(4, v1[0][3]);

  // Make the SB raise, BB call.
  tracker_.Update(10, 10, 0);
  const auto& v2 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(1, v2.size());
  EXPECT_EQ(1, v2[0][0]);
  EXPECT_EQ(2, v2[0][1]);
  EXPECT_EQ(1, v2[0][2]);
  EXPECT_EQ(4, v2[0][3]);
  EXPECT_EQ(8, v2[0][4]);
  EXPECT_EQ(4, v2[0][5]);

  // BB checks the flop.
  tracker_.Update(10, 10, 3);
  tracker_.Print();
  const auto& v3 = tracker_.History();
  EXPECT_EQ(2, v3.size());
  EXPECT_EQ(0, v3[1][0]);

  // SB bets 7, BB raises to 13.
  tracker_.Update(17, 23, 3);
  const auto& v4 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(0, v4[1][0]);
  EXPECT_EQ(7, v4[1][1]);
  EXPECT_EQ(13, v4[1][2]);

  // SB calls, done with flop, go to turn, BB bets 20.
  tracker_.Update(23, 43, 4);
  tracker_.Print();
  const auto& v5 = tracker_.History();
  EXPECT_EQ(3, v5.size());
  EXPECT_EQ(6, v5[1][3]);
  EXPECT_EQ(20, v5[2][0]);

  // SB raises, BB raises.
  tracker_.Update(50, 60, 4);
  tracker_.Print();
  const auto& v6 = tracker_.History();
  EXPECT_EQ(3, v6.size());
  EXPECT_EQ(20, v6[2][0]);
  EXPECT_EQ(27, v6[2][1]);
  EXPECT_EQ(17, v6[2][2]);

  // SB calls turn, go to river, BB checks.
  tracker_.Update(60, 60, 5);
  tracker_.Print();

  tracker_.Update(60, 60, 5);
  tracker_.Print();
  const auto& v7 = tracker_.History();
  EXPECT_EQ(4, v7.size());
  EXPECT_EQ(0, v7[3][0]);
  EXPECT_EQ(0, v7[3][1]);
  EXPECT_EQ(2, v7[3].size());
}


TEST(HistoryTrackerTest, testUpdateBb) {
  HistoryTracker tracker_(true);

  // Initialization - small and big blinds paid out.
  const auto& v0 = tracker_.History();
  // EXPECT_EQ(1, v0[0]);
  // EXPECT_EQ(2, v0[1]);

  // SB called.
  tracker_.Update(2, 2, 0);
  const auto& v1 = tracker_.History();
  // EXPECT_EQ(1, v1[0]);
  // EXPECT_EQ(2, v1[1]);
  // EXPECT_EQ(1, v1[2]);
  tracker_.Print();

  // BB bet and SB raised.
  tracker_.Update(10, 20, 0);
  const auto& v2 = tracker_.History();
  // EXPECT_EQ(1, v2[0]);
  // EXPECT_EQ(2, v2[1]);
  // EXPECT_EQ(1, v2[2]);
  // EXPECT_EQ(8, v2[3]);
  // EXPECT_EQ(18, v2[4]);
  tracker_.Print();

  // BB called on preflop and then BB checks the flop.
  tracker_.Update(20, 20, 3);
  const auto& v3 = tracker_.History();
  // EXPECT_EQ(10, v3[5]);
  // EXPECT_EQ(0, v3[8]);
  tracker_.Print();
}

TEST(HistoryTrackerTest, test_01) {
  // B posts the blind of 1
  // A posts the blind of 2
  // B dealt Th Ts [9h 9s]
  // A dealt 5c 5d [4c 4d]
  // B calls
  // A checks
  // Flop 8h Ad Qs [3h Kd 8s], B (2), A (2)
  // A checks
  // B bets 2
  // A calls
  // Turn 8h Ad Qs 9s [3h Kd 8s As], B (4), A (4)
  // A checks
  // B checks
  // River 8h Ad Qs 9s 4d [3h Kd 8s As 5d], B (4), A (4)
  // A checks
  // B bets 2
  // A calls
  // B shows Th Ts [9h 9s]
  // A shows 5c 5d [4c 4d]
  // B awarded 6
  // A awarded -6
  HistoryTracker tracker(true);
  tracker.Update(2, 2, 0);
  tracker.Update(2, 2, 3);
  tracker.Update(2, 4, 3);
  tracker.Print();
}
