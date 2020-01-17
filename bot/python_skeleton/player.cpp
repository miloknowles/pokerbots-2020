#include <iostream>

#include "player.hpp"

namespace pb {

static int MakeRelativeBet(const float frac, const int pot_size, const int min_raise, const int max_raise) {
  const int amt = static_cast<int>(frac * static_cast<float>(pot_size));
  const int clamped = std::min(max_raise, std::max(min_raise, amt));
  // printf("BET: frac=%f pot=%d min=%d max=%d\n", frac, pot_size, min_raise, max_raise);
  return clamped;
}

/**
 * Called when a new game starts. Called exactly once.
 */
Player::Player() {}

/**
 * Called when a new round starts. Called NUM_ROUNDS times.
 *
 * @param game_state Pointer to the GameState object.
 * @param round_state Pointer to the RoundState object.
 * @param active Your player's index.
 */
void Player::handle_new_round(GameState* game_state, RoundState* round_state, int active) {
  //int my_bankroll = game_state->bankroll;  // the total number of chips you've gained or lost from the beginning of the game to the start of this round
  float game_clock = game_state->game_clock;  // the total number of seconds your bot has left to play this game
  int round_num = game_state->round_num;  // the round number from 1 to NUM_ROUNDS
  //std::array<std::string, 2> my_cards = round_state->hands[active];  // your cards
  
  bool big_blind = static_cast<bool>(active);

  printf("\n================== NEW ROUND: %d ==================\n", round_num);
  std::cout << "*** TIME REMAINING: " << game_clock << std::endl;
  printf("*** Big blind: %d\n", big_blind);

  street_ev_.clear();

  history_ = HistoryTracker(big_blind);
}

/**
 * Called when a round ends. Called NUM_ROUNDS times.
 *
 * @param game_state Pointer to the GameState object.
 * @param terminal_state Pointer to the TerminalState object.
 * @param active Your player's index.
 */
void Player::handle_round_over(GameState* game_state, TerminalState* terminal_state, int active) {
  const int my_delta = terminal_state->deltas[active];  // your bankroll change from this round
  RoundState* previous_state = (RoundState*) terminal_state->previous_state;  // RoundState before payoffs
  const int street = previous_state->street;  // 0, 3, 4, or 5 representing when this round ended
  const int round_num = game_state->round_num;
  const int my_stack = previous_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = previous_state->stacks[1-active];  // the number of chips your opponent has remaining
  const int my_contribution = STARTING_STACK - my_stack;  // the number of chips you have contributed to the pot
  const int opp_contribution = STARTING_STACK - opp_stack;  // the number of chips your opponent has contributed to the pot
  history_.Update(my_contribution, opp_contribution, street);

  const std::array<std::string, 2> my_cards = previous_state->hands[active];  // your cards
  const std::array<std::string, 2> opp_cards = previous_state->hands[1-active];  // opponent's cards or "" if not revealed
  const std::array<std::string, 5> board_cards = previous_state->deck;

  const std::string win_hand = my_delta >= 0 ? my_cards[0] + my_cards[1] : opp_cards[0] + opp_cards[1];
  const std::string lose_hand = my_delta >= 0 ? opp_cards[0] + opp_cards[1] : my_cards[0] + my_cards[1];
  std::string board = "";
  for (const std::string s : board_cards) {
    board += s;
  }

  // Print out the profiling results on the last round.
  if (round_num == 1000) {
    pf_.Profile();
  }

  const bool did_see_showdown = lose_hand.size() == 4 && win_hand.size() == 4;
  if (did_see_showdown) {
    ++num_showdowns_seen_;

    const bool did_converge = pf_.Unique() == 1;

    if (pf_.Nonzero() <= 0 || did_converge) {
      if (pf_.Nonzero() <= 0) {
        std::cout << "FAILURE: PermutationFilter particles all died" << std::endl;
      } else {
        std::cout << "SUCCESS: Permutation filter converged" << std::endl;
      }
      return;
    }

    const ShowdownResult result(win_hand, lose_hand, board);
    pf_.Update(result);
    std::cout << "[ROUNDOVER] Updated with showdown result" << std::endl;
    std::cout << "[ROUNDOVER] Particle filter nonzero = " << pf_.Nonzero() << std::endl;
  }

  std::cout << "\n[ROUNDOVER] Final history:" << std::endl;
  history_.Print();
  std::cout << std::endl;
}

/**
 * Where the magic happens - your code should implement this function.
 * Called any time the engine needs an action from your bot.
 *
 * @param game_state Pointer to the GameState object.
 * @param round_state Pointer to the RoundState object.
 * @param active Your player's index.
 * @return Your action.
 */
Action Player::get_action(GameState* game_state, RoundState* round_state, int active) {
  const int round_num = game_state->round_num;
  const int legal_actions = round_state->legal_actions();  // mask representing the actions you are allowed to take
  const int street = round_state->street;  // 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
  const std::array<std::string, 2> my_cards = round_state->hands[active];  // your cards
  const std::array<std::string, 5> board_cards = round_state->deck;  // the board cards
  const int my_pip = round_state->pips[active];  // the number of chips you have contributed to the pot this round of betting
  const int opp_pip = round_state->pips[1-active];  // the number of chips your opponent has contributed to the pot this round of betting
  const int my_stack = round_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = round_state->stacks[1-active];  // the number of chips your opponent has remaining
  const int continue_cost = opp_pip - my_pip;  // the number of chips needed to stay in the pot
  const int my_contribution = STARTING_STACK - my_stack;  // the number of chips you have contributed to the pot
  const int opp_contribution = STARTING_STACK - opp_stack;  // the number of chips your opponent has contributed to the pot

  history_.Update(my_contribution, opp_contribution, street);

  // Check fold if no particles left.
  if (pf_.Nonzero() <= 0) {
    std::cout << "[GETACTION] [FAILURE] Particle filter empty, check-folding" << std::endl;
    return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
  }

  const bool did_converge = (num_showdowns_seen_ > num_showdowns_converge_) && pf_.Unique() < 10;
  printf("[GETACTION] Did converge? %d (unique=%d)\n", did_converge, pf_.Unique());

  // If EV hasn't been computed for this street, do it here.
  if (street_ev_.count(street) == 0) {
    std::string board_str;
    for (int i = 0; i < street; ++i) {
      board_str += board_cards[i];
    }
    // If converged, only sample ONE permutation.
    const int nsamples = did_converge ? 1 : compute_ev_samples_;
    const float ev_this_street = pf_.ComputeEvRandom(
        my_cards[0] + my_cards[1], board_str, "", nsamples, compute_ev_iters_);
    street_ev_[street] = ev_this_street;
  }
  const float EV = street_ev_.at(street);

  const int pot_size = my_contribution + opp_contribution;
  printf("[GETACTION] round=%d | street=%d | ev=%f | pot_size=%d | continue_cost=%d\n",
      round_num, street, EV, pot_size, continue_cost);
  
  const int min_raise = round_state->raise_bounds()[0];
  const int max_raise = round_state->raise_bounds()[1];

  const bool is_bb = static_cast<bool>(active);
  if (street == 0) {
    return HandleActionPreflop(EV, round_num, street, pot_size, continue_cost, legal_actions,
                               min_raise, max_raise, my_contribution, opp_contribution, is_bb);
  } else if (street == 3) {
    return HandleActionFlop(EV, round_num, street, pot_size, continue_cost, legal_actions,
                            min_raise, max_raise, my_contribution, opp_contribution, is_bb);
  } else {
    return HandleActionTurn(EV, round_num, street, pot_size, continue_cost, legal_actions,
                            min_raise, max_raise, my_contribution, opp_contribution, is_bb);
  }
}


Action Player::HandleActionPreflop(float EV, int round_num, int street, int pot_size,
                                   int continue_cost, int legal_actions, int min_raise,
                                   int max_raise, int my_contribution, int opp_contribution,
                                   bool is_big_blind) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool must_pay_to_continue = continue_cost > 0;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  const bool is_our_first_action = (is_big_blind && my_contribution == BIG_BLIND) || (!is_big_blind && my_contribution == SMALL_BLIND);

  if (is_our_first_action) {
    // CASE 1: First action and we are SMALL.
    if (!is_big_blind) {
      printf("First action for SB, ev=%f\n", EV);
      assert(continue_cost > 0);
      if (EV >= 0.60 && raise_is_allowed) {
        return RaiseAction(2 * BIG_BLIND);
      } else if (EV >= 0.70 && raise_is_allowed) {
        return RaiseAction(4 * BIG_BLIND);
      } else if (EV >= 0.80 && raise_is_allowed) {
        return RaiseAction(6 * BIG_BLIND);
      } else {
        return FoldAction();
      }
    // CASE 2: First action and we are BIG.
    } else {
      printf("First action for BB, ev=%f\n", EV);
      const bool other_player_did_raise = (continue_cost > 0);

      // If other player DID raise, consider reraising.
      if (other_player_did_raise) {
        const int pot_after_call = 2 * opp_contribution;
        const int equity = EV * pot_after_call;
        if (equity > 3 * continue_cost) {
          return RaiseAction(3 * BIG_BLIND);
        } else if (equity > 2 * continue_cost) {
          return CallAction();
        } else {
          return FoldAction();
        }
      
      // Other player called but didn't raise.
      } else {
        // Try to make the other player fold.
        if (EV >= 0.75 && raise_is_allowed) {
          return RaiseAction(2 * BIG_BLIND);
        } else if (EV > 0.90 && raise_is_allowed) {
          return RaiseAction(4 * BIG_BLIND);
        } else {
          return CheckAction();
        }
      }
    }
  
  // Not our first action, limit the number of re-raises we'll do.
  } else {
    // const int num_betting_rounds = next_action_idx_ / 2;
    const int num_betting_rounds = history_.NumBettingRounds();
    printf("Num bettings rounds so far: %d\n", num_betting_rounds);

    // CASE 3: Not our first action, must pay to continue.
    const bool other_player_did_raise = (continue_cost > 0);
    if (other_player_did_raise) {
      printf("Not first action, other player RAISED\n");
      const int pot_after_call = 2 * opp_contribution;
      const int equity = EV * pot_after_call;
      
      // Raise again only if pot odds are really good.
      if (equity > 5 * continue_cost) {
        return RaiseAction(2 * BIG_BLIND);
    
      // Otherwise, stay in it if it's barely worth it.
      } else if (equity > continue_cost) {
        return CallAction();
      } else {
        return FoldAction();
      }
    
    // CASE 4: Not our first action, don't need to pay to continue.
    } else {
      printf("Not first action, other player CHECKED/CALLED\n");
      // const int num_betting_rounds = next_action_idx_ / 2;
      const int num_betting_rounds = history_.NumBettingRounds();
      if (num_betting_rounds > 3) {
        return CheckAction();
      } else {
        // Try to make the other player fold.
        if (EV >= 0.75 && raise_is_allowed) {
          return RaiseAction(2 * BIG_BLIND);
        } else if (EV > 0.90 && raise_is_allowed) {
          return RaiseAction(4 * BIG_BLIND);
        } else {
          return CheckAction();
        }
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

Action Player::HandleActionFlop(float EV, int round_num, int street, int pot_size,
                                int continue_cost, int legal_actions, int min_raise,
                                int max_raise, int my_contribution, int opp_contribution,
                                bool is_big_blind) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;
  // printf("check_is_allowed=%d | raise_is_allowed=%d\n", check_is_allowed, raise_is_allowed);

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.6 || !raise_is_allowed) {
      std::cout << "[CONVERGED] (CheckAllowed) EV < 0.6 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If EV is pretty good (60-80%), do a pot raise.
    } else if (EV <= 0.8) {
      const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
      std::cout << "[CONVERGED] (CheckAllowed) EV <= 0.8 ==> PotRaise" << std::endl;
      return RaiseAction(raise_amt);

    // If EV above 0.8, do two pot raise.
    } else {
      const int raise_amt = MakeRelativeBet(2.0, pot_size, min_raise, max_raise);
      std::cout << "[CONVERGED] (CheckAllowed) EV above 0.8 ==> TwoPotRaise" << std::endl;
      return RaiseAction(raise_amt);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const int equity = EV * pot_after_call;
    printf("[CONVERGED] Call equity | equity=%d | pot_after_call=%d | continue_cost=%d\n", equity, pot_after_call, continue_cost);

    // Not worth it to continue.
    if (equity < (1.8 * continue_cost)) {
      std::cout << "[CONVERGED] (CallRequired) not worth it ==> FoldAction" << std::endl;
      return FoldAction();

    // Worth it - do bet sizing.
    } else {
      if (EV >= 0.7) {
        const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
        std::cout << "[CONVERGED] (CallRequired) worth it, EV >= 0.7 ==> PotRaise" << std::endl;
        return raise_is_allowed ? RaiseAction(raise_amt) : CallAction();
      } else if (EV >= 0.9) {
        const int raise_amt = MakeRelativeBet(2.0, pot_size, min_raise, max_raise);
        std::cout << "[CONVERGED] (CallRequired) worth it, EV >= 0.9 ==> TwoPotRaise" << std::endl;
        return raise_is_allowed ? RaiseAction(raise_amt) : CallAction();
      } else {
        std::cout << "[CONVERGED] (CallRequired) worth it, but EV not >= 0.7 ==> CallAction" << std::endl;
        return CallAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

Action Player::HandleActionTurn(float EV, int round_num, int street, int pot_size,
                                int continue_cost, int legal_actions, int min_raise,
                                int max_raise, int my_contribution, int opp_contribution,
                                bool is_big_blind) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;
  // printf("check_is_allowed=%d | raise_is_allowed=%d\n", check_is_allowed, raise_is_allowed);

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.6 || !raise_is_allowed) {
      std::cout << "[CONVERGED] (CheckAllowed) EV < 0.6 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If EV is pretty good (60-80%), do a pot raise.
    } else if (EV <= 0.8) {
      const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
      std::cout << "[CONVERGED] (CheckAllowed) EV <= 0.8 ==> PotRaise" << std::endl;
      return RaiseAction(raise_amt);

    // If EV above 0.8, do two pot raise.
    } else {
      const int raise_amt = MakeRelativeBet(2.0, pot_size, min_raise, max_raise);
      std::cout << "[CONVERGED] (CheckAllowed) EV above 0.8 ==> TwoPotRaise" << std::endl;
      return RaiseAction(raise_amt);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const int equity = EV * pot_after_call;
    printf("[CONVERGED] Call equity | equity=%d | pot_after_call=%d | continue_cost=%d\n", equity, pot_after_call, continue_cost);

    // Not worth it to continue.
    if (equity < (1.8 * continue_cost)) {
      std::cout << "[CONVERGED] (CallRequired) not worth it ==> FoldAction" << std::endl;
      return FoldAction();

    // Worth it - do bet sizing.
    } else {
      if (EV >= 0.7) {
        const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
        std::cout << "[CONVERGED] (CallRequired) worth it, EV >= 0.7 ==> PotRaise" << std::endl;
        return raise_is_allowed ? RaiseAction(raise_amt) : CallAction();
      } else if (EV >= 0.9) {
        const int raise_amt = MakeRelativeBet(2.0, pot_size, min_raise, max_raise);
        std::cout << "[CONVERGED] (CallRequired) worth it, EV >= 0.9 ==> TwoPotRaise" << std::endl;
        return raise_is_allowed ? RaiseAction(raise_amt) : CallAction();
      } else {
        std::cout << "[CONVERGED] (CallRequired) worth it, but EV not >= 0.7 ==> CallAction" << std::endl;
        return CallAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

}
