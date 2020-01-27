#include "./gtest/gtest.h"

#include "history_tracker.hpp"
#include "engine_modified.hpp"
#include "cfr.hpp"

using namespace pb;
using namespace cfr;

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

  EvInfoSet infoset = MakeInfoSet(tracker_, 0, true, 0.83, 0);
  infoset.Print();

  // Make the SB call, BB bet.
  tracker_.Update(2, 6, 0);
  const auto& v1 = tracker_.History();
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 0);
  infoset.Print();
  EXPECT_EQ(1, v1.size());
  EXPECT_EQ(1, v1[0][0]);
  EXPECT_EQ(2, v1[0][1]);
  EXPECT_EQ(1, v1[0][2]);
  EXPECT_EQ(4, v1[0][3]);

  // Make the SB raise, BB call.
  tracker_.Update(10, 10, 0);
  const auto& v2 = tracker_.History();
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 0);
  infoset.Print();
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
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 3);
  infoset.Print();
  const auto& v3 = tracker_.History();
  EXPECT_EQ(2, v3.size());
  EXPECT_EQ(0, v3[1][0]);

  // SB bets 7, BB raises to 13.
  tracker_.Update(17, 23, 3);
  const auto& v4 = tracker_.History();
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 3);
  infoset.Print();
  EXPECT_EQ(0, v4[1][0]);
  EXPECT_EQ(7, v4[1][1]);
  EXPECT_EQ(13, v4[1][2]);

  // SB calls, done with flop, go to turn, BB bets 20.
  tracker_.Update(23, 43, 4);
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 4);
  infoset.Print();
  const auto& v5 = tracker_.History();
  EXPECT_EQ(3, v5.size());
  EXPECT_EQ(6, v5[1][3]);
  EXPECT_EQ(20, v5[2][0]);

  // SB raises, BB raises.
  tracker_.Update(50, 60, 4);
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 4);
  infoset.Print();
  const auto& v6 = tracker_.History();
  EXPECT_EQ(3, v6.size());
  EXPECT_EQ(20, v6[2][0]);
  EXPECT_EQ(27, v6[2][1]);
  EXPECT_EQ(17, v6[2][2]);

  // SB calls turn, go to river, BB checks.
  tracker_.Update(60, 60, 5);
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 5);
  infoset.Print();

  const auto& bucket = BucketInfoSetSmall(infoset);
  std::cout << BucketSmallJoin(bucket) << std::endl;

  tracker_.Update(60, 60, 5);
  tracker_.Print();
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 5);
  infoset.Print();
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
  EXPECT_EQ(1, v0[0][0]);
  EXPECT_EQ(2, v0[0][1]);

  // SB calls, ending preflop.
  tracker_.Update(2, 2, 3);
  const auto& v1 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(1, v1[0][0]);
  EXPECT_EQ(2, v1[0][1]);
  EXPECT_EQ(1, v1[0][2]);

  // Flop: BB bet and SB raised.
  tracker_.Update(10, 20, 3);
  const auto& v2 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(1, v2[0][0]);
  EXPECT_EQ(2, v2[0][1]);
  EXPECT_EQ(1, v2[0][2]);
  EXPECT_EQ(8, v2[1][0]);
  EXPECT_EQ(18, v2[1][1]);

  // BB called, ending flop. This is before BB's first action on turn.
  tracker_.Update(20, 20, 4);
  const auto& v3 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(10, v3[1][2]);

  // Turn: BB checks, SB checks.
  tracker_.Update(20, 20, 4);
  const auto& v4 = tracker_.History();
  tracker_.Print();
  EXPECT_EQ(0, v4[2][0]);
  EXPECT_EQ(0, v4[2][1]);
}


TEST(InfoSetTest, testBetHistoryWraps) {
  HistoryTracker tracker_(true);

  // SB raises.
  tracker_.Update(2, 4, 0);

  // BB raises, SB re-raises.
  tracker_.Update(8, 12, 0);

  // BB reraises, SB re-raises.
  tracker_.Update(16, 20, 0);

  // BB reraises, SB re-raises.
  tracker_.Update(24, 30, 0);

  // BB calls, ending preflop.
  tracker_.Update(30, 30, 3);

  const EvInfoSet info = MakeInfoSet(tracker_, 1, false, 0.73, 3);
  info.Print();

  tracker_.Update(40, 50, 3);
  tracker_.Update(60, 70, 3);
  tracker_.Update(80, 90, 3);
  tracker_.Update(100, 120, 3);
  const EvInfoSet info2 = MakeInfoSet(tracker_, 1, false, 0.73, 3);
  info2.Print();

  tracker_.Update(120, 120, 4);
  const EvInfoSet info3 = MakeInfoSet(tracker_, 1, false, 0.73, 4);
  info3.Print();  
}


TEST(HistoryTrackerTest, test01) {
  // A posts the blind of 1
  // B posts the blind of 2
  // A dealt 9c 8c [Ac Jc]
  // B dealt 2d 4c [2d 5c]
  // A raises to 6
  // B calls
  // Flop 3d 3h 6s [4d 4h 9s], A (6), B (6)
  // B checks
  // A bets 12
  // B raises to 24
  // A calls
  // Turn 3d 3h 6s Qh [4d 4h 9s Kh], A (30), B (30)
  // B checks
  // A checks
  // River 3d 3h 6s Qh 4h [4d 4h 9s Kh 5h], A (30), B (30)
  // B bets 2
  // A folds
  // A awarded -30
  // B awarded 30

  // TODO: fix this!!!
  HistoryTracker tracker_(false);
  tracker_.Update(1, 2, 0);

  // A raises, B calls, B checks.
  tracker_.Update(6, 6, 3);
  EvInfoSet infoset = MakeInfoSet(tracker_, 0, true, 0.83, 3);
  std::string bucket = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << "A raises, B calls, B checks on flop" << std::endl;
  std::cout << bucket << std::endl;

  // A raises, B raises.
  tracker_.Update(12, 24, 3);
  infoset = MakeInfoSet(tracker_, 0, true, 0.83, 3);
  infoset.Print();
  bucket = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << "A raises, B raises" << std::endl;
  std::cout << bucket << std::endl;
}
