#include "./gtest/gtest.h"

#include "engine_modified.hpp"
#include "infoset.hpp"
#include "cfr.hpp"

using namespace pb;


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

  // SB calls.
  RoundState* next = dynamic_cast<RoundState*>(round_state.proceed(CallAction()));
  assert(next != nullptr);

  EXPECT_EQ(0, next->street);
  EXPECT_EQ(2, next->button);
  PrintFlexHistory(next->bet_history);

  bool can_check = CHECK_ACTION_TYPE & next->legal_actions();
  bool can_call = CALL_ACTION_TYPE & next->legal_actions();
  bool can_fold = FOLD_ACTION_TYPE & next->legal_actions();
  bool can_raise = RAISE_ACTION_TYPE & next->legal_actions();

  EXPECT_TRUE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_FALSE(can_fold);
  EXPECT_FALSE(can_call);

  // BB raises.
  next = dynamic_cast<RoundState*>(next->proceed(RaiseAction(4)));

  EXPECT_EQ(0, next->street);
  EXPECT_EQ(3, next->button);
  PrintFlexHistory(next->bet_history);

  can_check = CHECK_ACTION_TYPE & next->legal_actions();
  can_call = CALL_ACTION_TYPE & next->legal_actions();
  can_fold = FOLD_ACTION_TYPE & next->legal_actions();
  can_raise = RAISE_ACTION_TYPE & next->legal_actions();

  EXPECT_FALSE(can_check);
  EXPECT_TRUE(can_raise);
  EXPECT_TRUE(can_fold);
  EXPECT_TRUE(can_call);

  // SB re-raises to 8.
  next = dynamic_cast<RoundState*>(next->proceed(RaiseAction(8)));
  EXPECT_EQ(0, next->street);
  EXPECT_EQ(4, next->button);
  PrintFlexHistory(next->bet_history);

  // BB calls.
  next = dynamic_cast<RoundState*>(next->proceed(CallAction()));
  EXPECT_EQ(3, next->street);
  EXPECT_EQ(0, next->button);
  PrintFlexHistory(next->bet_history);

  // BB checks, SB checks.
  std::cout << "First check on FLOP" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(CheckAction()));
  EXPECT_EQ(3, next->street);
  EXPECT_EQ(1, next->button);
  PrintFlexHistory(next->bet_history);
  std::cout << "Second check on FLOP" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(CheckAction()));
  EXPECT_EQ(4, next->street);
  EXPECT_EQ(0, next->button);
  PrintFlexHistory(next->bet_history);

  // BB checks, SB raises.
  std::cout << "First check on TURN" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(CheckAction()));
  EXPECT_EQ(4, next->street);
  EXPECT_EQ(1, next->button);
  PrintFlexHistory(next->bet_history);
  std::cout << "SB raises on TURN" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(RaiseAction(20)));
  EXPECT_EQ(4, next->street);
  EXPECT_EQ(2, next->button);
  PrintFlexHistory(next->bet_history);

  // BB re-raises, SB calls.
  std::cout << "BB reraises on TURN" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(RaiseAction(40)));
  EXPECT_EQ(4, next->street);
  EXPECT_EQ(3, next->button);
  PrintFlexHistory(next->bet_history);
  std::cout << "SB calls on TURN" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(CallAction()));
  EXPECT_EQ(5, next->street);
  EXPECT_EQ(0, next->button);
  PrintFlexHistory(next->bet_history);

  // Double check on RIVER.
  std::cout << "First check on RIVER" << std::endl;
  next = dynamic_cast<RoundState*>(next->proceed(CheckAction()));
  EXPECT_EQ(5, next->street);
  EXPECT_EQ(1, next->button);
  PrintFlexHistory(next->bet_history);
  std::cout << "Second check on RIVER" << std::endl;
  TerminalState* end = dynamic_cast<TerminalState*>(next->proceed(CheckAction()));
  EXPECT_TRUE(end != nullptr);

  PrintFlexHistory(dynamic_cast<RoundState*>(end->previous_state)->bet_history);
  std::cout << end->deltas[0] << " " << end->deltas[1] << std::endl;
}
