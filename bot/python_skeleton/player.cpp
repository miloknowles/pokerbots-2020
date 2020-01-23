#include <iostream>

#include "player.hpp"

namespace pb {


static bool CanCheckFoldRemainder(const int delta, const int round_num) {
  return delta > (1.5f * (1000.0f - static_cast<float>(round_num)) + 1);
}


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
  int my_bankroll = game_state->bankroll;  // the total number of chips you've gained or lost from the beginning of the game to the start of this round
  float game_clock = game_state->game_clock;  // the total number of seconds your bot has left to play this game
  int round_num = game_state->round_num;  // the round number from 1 to NUM_ROUNDS
  //std::array<std::string, 2> my_cards = round_state->hands[active];  // your cards
  
  bool big_blind = static_cast<bool>(active);
  check_fold_mode_ = CanCheckFoldRemainder(my_bankroll, round_num);

  printf("\n================== NEW ROUND: %d ==================\n", round_num);
  std::cout << "*** TIME REMAINING: " << game_clock << std::endl;
  printf("*** Big blind: %d\n", big_blind);

  street_ev_.clear();
  current_street_ = -1;
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

  if (street != current_street_) {
    current_street_ = street;
    if (current_street_ == 0) {
      std::cout << "*** PREFLOP ***" << std::endl;
    } else if (current_street_ == 3) {
      std::cout << "*** FLOP ***" << std::endl;
    } else if (current_street_ == 4) {
      std::cout << "*** TURN ***" << std::endl;
    } else {
      std::cout << "*** RIVER ***" << std::endl;
    }
  }
  history_.Update(my_contribution, opp_contribution, street);

  if (check_fold_mode_) {
    std::cout << "*** [WIN] CHECK-FOLD MODE ACTIVATED ***" << std::endl;
    const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
    return check_is_allowed ? CheckAction() : FoldAction();
  }

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
  printf("[GETACTION] round=%d | street=%d | ev=%f | pot_size=%d | continue_cost=%d |\n",
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

  const bool is_our_first_action = (is_big_blind && my_contribution == BIG_BLIND) ||
                                   (!is_big_blind && my_contribution == SMALL_BLIND);
  
  // const int num_betting_rounds = history_.TotalBets(0).first;
  // printf("Num bettings rounds so far: %d\n", num_betting_rounds);

  if (is_our_first_action) {
    // CASE 1: First action and we are SMALLBLIND.
    if (!is_big_blind) {
      std::cout << "PREFLOP_1: We are SB, first action" << std::endl;
      assert(continue_cost > 0);
      if (EV >= 0.80 && raise_is_allowed) {
        std::cout << "1A: great EV, 8BB raise" << std::endl;
        return RaiseAction(8 * BIG_BLIND);
      } else if (EV >= 0.70 && raise_is_allowed) {
        std::cout << "1B: pretty good EV, 6BB raise" << std::endl;
        return RaiseAction(6 * BIG_BLIND);
      } else if (EV >= 0.50 && raise_is_allowed) {
        std::cout << "1C: okay EV, 10BB raise to try to push them out" << std::endl;
        return RaiseAction(6 * BIG_BLIND);
      } else if (EV >= 50) {
        return CallAction();
      } else {
        std::cout << "EV not good enough to call BB, folding" << std::endl;
        return FoldAction();
      }
    // CASE 2: First action and we are BIG.
    } else {
      std::cout << "PREFLOP_2: first action and we are BB" << std::endl;
      const bool other_player_did_raise = (continue_cost > 0);

      // If other player DID raise, consider reraising.
      if (other_player_did_raise) {
        std::cout << "the opponent raised on their first action" << std::endl;
        const int pot_after_call = 2 * opp_contribution;
        const float equity = EV * static_cast<float>(pot_after_call);
        if (EV > 0.80) {
          std::cout << "2A: really high chance of winning, doing 6BB re-raise" << std::endl;
          return RaiseAction(6 * BIG_BLIND);
        } else if (EV > 0.70) {
          std::cout << "2B: pretty good EV, doing 4BB re-raise" << std::endl;
          return RaiseAction(4 * BIG_BLIND);
        } else if (equity > 0.60) {
          std::cout << "2C: OK ev, calling" << std::endl;
          return CallAction();
        } else {
          std::cout << "2D: other player raised but equity not high enough, folding" << std::endl;
          return FoldAction();
        }
      
      // Other player called but didn't raise.
      } else {
        std::cout << "We are BB, opponent called their first action." << std::endl;
        // Try to make the other player fold.
        if (EV >= 0.65 && raise_is_allowed) {
          std::cout << "2D: okay EV, trying to push out other player before flop, raising 8BB" << std::endl;
          return RaiseAction(6 * BIG_BLIND);
        } else {
          std::cout << "2G: EV not high enough to raise, checking" << std::endl;
          return CheckAction();
        }
      }
    }
  
  // Not our first action, limit the number of re-raises we'll do.
  } else {
    const bool other_player_did_raise = (continue_cost > 0);
  
    // CASE 3: Not our first action, must pay to continue.
    if (other_player_did_raise) {
      std::cout << "PREFLOP_3: not our first action, other player raised" << std::endl;
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);
      
      // Raise again only if pot odds are really good.
      if (equity > 8.0f * continue_cost) {
        std::cout << "3A: great pot odds (8:1), 6BB raise" << std::endl;
        return raise_is_allowed ? RaiseAction(6 * BIG_BLIND) : CallAction();
    
      // Otherwise, stay in it if it's barely worth it.
      } else if (equity > 5.0f * continue_cost) {
        std::cout << "3B: good pot odds (5:1), raising 2BB" << std::endl;
        return raise_is_allowed ? RaiseAction(4 * BIG_BLIND) : CallAction();
      } else if (equity > 1.5f * continue_cost) {
        return CallAction();
      } else {
        std::cout << "3C: equity not good enough to continue, folding" << std::endl;
        return FoldAction();
      }
    
    // CASE 4: Not our first action, don't need to pay to continue.
    } else {
      std::cout << "PREFLOP_4: not our first action, call NOT required." << std::endl;

      // If we've already increased the pot 4 times, just check.
      // if (num_betting_rounds >= 4) {
      //   std::cout << "already done 4 betting actions on preflop, just checking" << std::endl;
      //   return CheckAction();
      // } else {
      // Try to make the other player fold.
      if (EV >= 0.80 && raise_is_allowed) {
        std::cout << "4A: great EV, doing 8BB bet" << std::endl;
        return RaiseAction(8 * BIG_BLIND);
      } else if (EV > 0.70 && raise_is_allowed) {
        std::cout << "4B: good EV, doing 4BB bet" << std::endl;
        return RaiseAction(4 * BIG_BLIND);
      } else {
        std::cout << "4C: EV not good enough to bet, just checking" << std::endl;
        return CheckAction();
      }
      // }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

Action Player::HandleActionFlop(float EV, int round_num, int street, int pot_size,
                                int continue_cost, int legal_actions, int min_raise,
                                int max_raise, int my_contribution, int opp_contribution,
                                bool is_big_blind) {
  std::cout << "Getting a FLOP action" << std::endl;
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  if (!is_big_blind) {
    // CASE 1: If the opponent checked the flop, they probably don't have anything too good.
    if (check_is_allowed) {
      // EV isn't very good, consider trying to fold out the opponent.
      if (EV < 0.5) {
        // i.e if EV is 30%, bluff it 30% of the time.
        const bool do_bluff_raise = real_(gen_) < EV;
        if (do_bluff_raise) {
          std::cout << "1A: opponent checked on the FLOP, randomly chose to BLUFF" << std::endl;
          const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
          return raise_is_allowed ? RaiseAction(amt) : CheckAction();
        } else {
          std::cout << "1B: opponent checked on the FLOP, randomly chose to CHECK" << std::endl;
          return CheckAction();
        }
      } else if (EV > 0.9) {
        const int amt = MakeRelativeBet(3.0f, pot_size, min_raise, max_raise);
        std::cout << "1C: really good EV, 3pot bet" << std::endl;
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else if (EV > 0.7) {
        std::cout << "1D: pretty good EV, pot bet" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else {
        std::cout << "1E: middle EV, check" << std::endl;
        return CheckAction();
      }
    
    // CASE 2: The opponent bet on the flop, they probably have something good?
    } else {
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);

      if (equity > 4*continue_cost) {
        std::cout << "2A: great pot odds, pot raise" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CallAction();
      } else if (equity > 1.3*continue_cost) {
        std::cout << "2B: pretty good pot odds, just call" << std::endl;
        return CallAction();
      } else {
        std::cout << "2C: not good pot odds, fold" << std::endl;
        return FoldAction();
      }
    }
  } else {
    // CASE 3: We are BB and this is the FIRST action.
    if (check_is_allowed) {
      if (EV > 0.8) {
        std::cout << "3A: great EV, 2pot raise" << std::endl;
        const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else if (EV > 0.7) {
        std::cout << "3B: good EV, pot raise" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else {
        return CheckAction();
      }
    
    // CASE 4: We are BB and this is not the FIRST action - opponent must have bet or raised.
    } else {
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);

      if (equity > 4*continue_cost) {
        std::cout << "4A: great pot odds (4:1), pot raise" << std::endl;
        const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CallAction();
      } else if (equity > 2*continue_cost) {
        std::cout << "4B: pretty good (2:1) pot odds, just call" << std::endl;
        return CallAction();
      } else {
        std::cout << "4C: not good pot odds, fold" << std::endl;
        return FoldAction();
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

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.75 || !raise_is_allowed) {
      std::cout << "1A: (CheckAllowed) EV < 0.7 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If EV is pretty good (70-85%), do a pot raise.
    } else if (EV <= 0.85) {
      const int raise_amt = MakeRelativeBet(0.5, pot_size, min_raise, max_raise);
      std::cout << "1B: (CheckAllowed) EV <= 0.8 ==> 1/2PotRaise" << std::endl;
      return RaiseAction(raise_amt);

    // If EV above 0.8, do pot raise.
    } else {
      const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
      std::cout << "1C: (CheckAllowed) EV above 0.8 ==> PotRaise" << std::endl;
      return RaiseAction(raise_amt);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const float equity = EV * static_cast<float>(pot_after_call);

    if (EV > 0.95) {
      std::cout << "2A: really high EV, 2pot raise" << std::endl;
      const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
      return raise_is_allowed ? RaiseAction(amt) : CallAction();
    } else if (EV > 0.80) {
      std::cout << "2B: pretty high EV, pot raise" << std::endl;
      const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
      return raise_is_allowed ? RaiseAction(amt) : CallAction();
    } else if (equity > 3*continue_cost) {
      std::cout << "2C: good enough pot odds (3:1), calling" << std::endl;
      return CallAction();
    } else {
      std::cout << "pot odds not good enough, folding" << std::endl;
      return FoldAction();
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

}
