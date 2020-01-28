#include "./gtest/gtest.h"

#include "engine_modified.hpp"
#include "infoset.hpp"
#include "cfr.hpp"

using namespace pb;
using namespace cfr;

// Make sure that random hands and board are generated.
TEST(CfrTest, testCreateNewRound) {
  for (int k = 0; k < 10; ++k) {
    const int sb_plyr_idx = k % 2;
    RoundState round_state = CreateNewRound(sb_plyr_idx);

    std::array<double, 2> reach_probabilities = { 1.0, 1.0 };
    PrecomputedEv precomputed_ev = MakePrecomputedEv(round_state);

    int rctr = 0;

    std::cout << round_state.hands[0][0] << round_state.hands[0][1] << std::endl;
    std::cout << round_state.hands[1][0] << round_state.hands[1][1] << std::endl;

    for (int i = 0; i < 5; ++i) {
      std::cout << round_state.deck[i] << " ";
    }
    std::cout << std::endl;
  }
}


TEST(CfrTest, testEngine) {
  const int sb_plyr_idx = 1;
  RoundState round_state = CreateNewRound(sb_plyr_idx);

  PrecomputedEv ev = MakePrecomputedEv(round_state);
  printf("Precomputed EV: PREFLOP[0]=%f PREFLOP[1]=%f\n", ev[0][0], ev[1][0]);
  printf("Precomputed EV: FLOP[0]=%f FLOP[1]=%f\n", ev[0][1], ev[1][1]);
  printf("Precomputed EV: TURN[0]=%f TURN[1]=%f\n", ev[0][2], ev[1][2]);
  printf("Precomputed EV: RIVER[0]=%f RIVER[1]=%f\n", ev[0][3], ev[1][3]);

  std::cout << "Should see initial config for SB" << std::endl;
  EvInfoSet infoset = MakeInfoSet(round_state, 1, true, ev);
  infoset.Print();

  // SB calls.
  RoundState next = round_state.proceed(CallAction());

  std::cout << "Should see SB call" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  infoset.Print();

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);

  bool can_check = CHECK_ACTION_TYPE & next.legal_actions();
  bool can_call = CALL_ACTION_TYPE & next.legal_actions();
  bool can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  bool can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_TRUE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_FALSE(can_fold);
  EXPECT_FALSE(can_call);

  // BB raises.
  next = next.proceed(RaiseAction(4));

  std::cout << "Should see SB call, BB raise" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  std::string b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);

  can_check = CHECK_ACTION_TYPE & next.legal_actions();
  can_call = CALL_ACTION_TYPE & next.legal_actions();
  can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_FALSE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_TRUE(can_fold);
  EXPECT_TRUE(can_call);

  // SB re-raises to 8.
  next = next.proceed(RaiseAction(8));
  EXPECT_EQ(0, next.street);
  EXPECT_EQ(4, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Should see SB call, BB raise, SB reraise" << std::endl;
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB calls.
  next = next.proceed(CallAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Start of flop" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB checks, SB raises.
  std::cout << "First check on TURN" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "SB raises on TURN" << std::endl;
  next = next.proceed(RaiseAction(20));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB re-raises, SB calls.
  std::cout << "BB reraises on TURN" << std::endl;
  next = next.proceed(RaiseAction(40));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "SB calls on TURN" << std::endl;
  next = next.proceed(CallAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // Double check on RIVER.
  std::cout << "First check on RIVER" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on RIVER" << std::endl;
  const RoundState& end = next.proceed(CheckAction());

  PrintFlexHistory(end.bet_history);
  std::cout << end.deltas[0] << " " << end.deltas[1] << std::endl;
}


TEST(CfrTest, testSbPlyrIdx0) {
  const int sb_plyr_idx = 0;
  RoundState round_state = CreateNewRound(sb_plyr_idx);

  PrecomputedEv ev = MakePrecomputedEv(round_state);
  printf("Precomputed EV: PREFLOP[0]=%f PREFLOP[1]=%f\n", ev[0][0], ev[1][0]);
  printf("Precomputed EV: FLOP[0]=%f FLOP[1]=%f\n", ev[0][1], ev[1][1]);
  printf("Precomputed EV: TURN[0]=%f TURN[1]=%f\n", ev[0][2], ev[1][2]);
  printf("Precomputed EV: RIVER[0]=%f RIVER[1]=%f\n", ev[0][3], ev[1][3]);

  std::cout << "Should see initial config for SB" << std::endl;
  EvInfoSet infoset = MakeInfoSet(round_state, 0, true, ev);
  infoset.Print();

  // SB calls.
  RoundState next = round_state.proceed(CallAction());

  std::cout << "Should see SB call" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  infoset.Print();

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);

  bool can_check = CHECK_ACTION_TYPE & next.legal_actions();
  bool can_call = CALL_ACTION_TYPE & next.legal_actions();
  bool can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  bool can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_TRUE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_FALSE(can_fold);
  EXPECT_FALSE(can_call);

  // BB raises.
  next = next.proceed(RaiseAction(4));

  std::cout << "Should see SB call, BB raise" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  std::string b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);

  can_check = CHECK_ACTION_TYPE & next.legal_actions();
  can_call = CALL_ACTION_TYPE & next.legal_actions();
  can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_FALSE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_TRUE(can_fold);
  EXPECT_TRUE(can_call);

  // SB re-raises to 8.
  next = next.proceed(RaiseAction(8));
  EXPECT_EQ(0, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Should see SB call, BB raise, SB reraise" << std::endl;
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB calls.
  next = next.proceed(CallAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Start of flop" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB checks, SB raises.
  std::cout << "First check on TURN" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "SB raises on TURN" << std::endl;
  next = next.proceed(RaiseAction(20));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // BB re-raises, SB calls.
  std::cout << "BB reraises on TURN" << std::endl;
  next = next.proceed(RaiseAction(40));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(4, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "SB calls on TURN" << std::endl;
  next = next.proceed(CallAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  // Double check on RIVER.
  std::cout << "First check on RIVER" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketSmallJoin(BucketInfoSetSmall(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on RIVER" << std::endl;
  const RoundState& end = next.proceed(CheckAction());

  PrintFlexHistory(end.bet_history);
  std::cout << end.deltas[0] << " " << end.deltas[1] << std::endl;
}


TEST(RegretMatchedStrategyTest, testSaveLoad) {
  RegretMatchedStrategy rm;

  for (int i = 0; i < 123; ++i) {
    const EvInfoSet infoset(0.123, FixedHistory(), i % 2, 5);
    rm.AddRegret(infoset, ActionRegrets());
  }

  rm.Save("./tmp_test_cfr_regrets.txt");
  rm.Load("./tmp_test_cfr_regrets.txt");
}


TEST(CfrTest, testMedium) {
  const int sb_plyr_idx = 0;
  RoundState round_state = CreateNewRound(sb_plyr_idx);

  PrecomputedEv ev = MakePrecomputedEv(round_state);
  printf("Precomputed EV: PREFLOP[0]=%f PREFLOP[1]=%f\n", ev[0][0], ev[1][0]);
  printf("Precomputed EV: FLOP[0]=%f FLOP[1]=%f\n", ev[0][1], ev[1][1]);
  printf("Precomputed EV: TURN[0]=%f TURN[1]=%f\n", ev[0][2], ev[1][2]);
  printf("Precomputed EV: RIVER[0]=%f RIVER[1]=%f\n", ev[0][3], ev[1][3]);

  std::cout << "Should see initial config for SB" << std::endl;
  EvInfoSet infoset = MakeInfoSet(round_state, 0, true, ev);
  infoset.Print();

  // SB calls.
  RoundState next = round_state.proceed(CallAction());

  std::cout << "Should see SB call" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  infoset.Print();

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);

  bool can_check = CHECK_ACTION_TYPE & next.legal_actions();
  bool can_call = CALL_ACTION_TYPE & next.legal_actions();
  bool can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  bool can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_TRUE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_FALSE(can_fold);
  EXPECT_FALSE(can_call);

  // BB raises.
  next = next.proceed(RaiseAction(4));

  std::cout << "Should see SB call, BB raise" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  std::string b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);

  can_check = CHECK_ACTION_TYPE & next.legal_actions();
  can_call = CALL_ACTION_TYPE & next.legal_actions();
  can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_FALSE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_TRUE(can_fold);
  EXPECT_TRUE(can_call);

  // SB re-raises to 8.
  next = next.proceed(RaiseAction(8));
  EXPECT_EQ(0, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Should see SB call, BB raise, SB reraise" << std::endl;
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB calls.
  next = next.proceed(CallAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Start of flop" << std::endl;
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB checks, SB raises.
  std::cout << "First check on TURN" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "SB raises on TURN" << std::endl;
  next = next.proceed(RaiseAction(20));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB re-raises, SB calls.
  std::cout << "BB reraises on TURN" << std::endl;
  next = next.proceed(RaiseAction(40));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(4, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "SB calls on TURN" << std::endl;
  next = next.proceed(CallAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // Double check on RIVER.
  std::cout << "First check on RIVER" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on RIVER" << std::endl;
  const RoundState& end = next.proceed(CheckAction());

  PrintFlexHistory(end.bet_history);
  std::cout << end.deltas[0] << " " << end.deltas[1] << std::endl;
}


TEST(CfrTest, testMedium2) {
  const int sb_plyr_idx = 1;
  RoundState round_state = CreateNewRound(sb_plyr_idx);

  PrecomputedEv ev = MakePrecomputedEv(round_state);
  printf("Precomputed EV: PREFLOP[0]=%f PREFLOP[1]=%f\n", ev[0][0], ev[1][0]);
  printf("Precomputed EV: FLOP[0]=%f FLOP[1]=%f\n", ev[0][1], ev[1][1]);
  printf("Precomputed EV: TURN[0]=%f TURN[1]=%f\n", ev[0][2], ev[1][2]);
  printf("Precomputed EV: RIVER[0]=%f RIVER[1]=%f\n", ev[0][3], ev[1][3]);

  std::cout << "Should see initial config for SB" << std::endl;
  EvInfoSet infoset = MakeInfoSet(round_state, 1, true, ev);
  infoset.Print();

  // SB calls.
  RoundState next = round_state.proceed(CallAction());

  std::cout << "Should see SB call" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  infoset.Print();

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);

  bool can_check = CHECK_ACTION_TYPE & next.legal_actions();
  bool can_call = CALL_ACTION_TYPE & next.legal_actions();
  bool can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  bool can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_TRUE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_FALSE(can_fold);
  EXPECT_FALSE(can_call);

  // BB raises.
  next = next.proceed(RaiseAction(4));

  std::cout << "Should see SB call, BB raise" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  std::string b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);

  can_check = CHECK_ACTION_TYPE & next.legal_actions();
  can_call = CALL_ACTION_TYPE & next.legal_actions();
  can_fold = FOLD_ACTION_TYPE & next.legal_actions();
  can_raise = RAISE_ACTION_TYPE & next.legal_actions();

  EXPECT_FALSE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_TRUE(can_fold);
  EXPECT_TRUE(can_call);

  // SB re-raises to 8.
  next = next.proceed(RaiseAction(8));
  EXPECT_EQ(0, next.street);
  EXPECT_EQ(4, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Should see SB call, BB raise, SB reraise" << std::endl;
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB calls.
  next = next.proceed(CallAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Start of flop" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB checks, SB raises.
  std::cout << "First check on TURN" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "SB raises on TURN" << std::endl;
  next = next.proceed(RaiseAction(20));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB re-raises, SB calls.
  std::cout << "BB reraises on TURN" << std::endl;
  next = next.proceed(RaiseAction(40));
  EXPECT_EQ(4, next.street);
  EXPECT_EQ(3, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "SB calls on TURN" << std::endl;
  next = next.proceed(CallAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // Double check on RIVER.
  std::cout << "First check on RIVER" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(5, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "Second check on RIVER" << std::endl;
  const RoundState& end = next.proceed(CheckAction());

  PrintFlexHistory(end.bet_history);
  std::cout << end.deltas[0] << " " << end.deltas[1] << std::endl;
}

TEST(CfrTest, testMedium3) {
  const int sb_plyr_idx = 1;
  RoundState round_state = CreateNewRound(sb_plyr_idx);

  PrecomputedEv ev = MakePrecomputedEv(round_state);

  std::cout << "Should see initial config for SB" << std::endl;
  EvInfoSet infoset = MakeInfoSet(round_state, 1, true, ev);
  infoset.Print();

  // SB calls.
  RoundState next = round_state.proceed(CallAction());

  std::cout << "Should see SB call" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  infoset.Print();

  EXPECT_EQ(0, next.street);
  EXPECT_EQ(2, next.button);
  PrintFlexHistory(next.bet_history);

  // BB checks.
  next = next.proceed(CheckAction());

  EXPECT_EQ(3, next.street);
  EXPECT_EQ(0, next.button);
  PrintFlexHistory(next.bet_history);

  std::cout << "Start of flop" << std::endl;
  infoset = MakeInfoSet(next, 0, false, ev);
  std::string b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = next.proceed(CheckAction());
  EXPECT_EQ(3, next.street);
  EXPECT_EQ(1, next.button);
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  next = next.proceed(RaiseAction(2));
  next = next.proceed(RaiseAction(4));
  next = next.proceed(RaiseAction(6));
  next = next.proceed(RaiseAction(8));

  std::cout << "BB check, SB raise, BB raise, SB raise, BB raise" << std::endl;
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 1, true, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "BB check, SB raise, BB raise, SB raise, BB raise, SB raise" << std::endl;
  next = next.proceed(RaiseAction(10));
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;

  std::cout << "BB check, SB raise, BB raise, SB raise, BB raise, SB raise, BB raise" << std::endl;
  next = next.proceed(RaiseAction(12));
  PrintFlexHistory(next.bet_history);
  infoset = MakeInfoSet(next, 0, false, ev);
  b = BucketJoin19(BucketInfoSetMedium(infoset));
  std::cout << b << std::endl;
}
