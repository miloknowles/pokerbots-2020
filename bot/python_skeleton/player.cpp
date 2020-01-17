#include <iostream>

#include "player.hpp"

namespace pb {

static int MakeRelativeBet(const float frac, const int pot_size, const int min_raise, const int max_raise) {
  const int amt = static_cast<int>(frac * static_cast<float>(pot_size));
  const int clamped = std::min(max_raise, std::max(min_raise, amt));
  // printf("BET: frac=%f pot=%d min=%d max=%d\n", frac, pot_size, min_raise, max_raise);
  return clamped;
}

static int GetStreet0123(const int street_sz) {
  return street_sz == 0 ? 0 : (street_sz - 2);
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

  // Reset history-related stuff.
  next_action_idx_ = 2;
  history_ = std::array<int, 4*kMaxActionsPerStreet>();
  history_[0] = 1;
  history_[1] = 2;
  contributions_[0] = big_blind ? 2 : 1;
  contributions_[1] = big_blind ? 1 : 2;
  prev_street_ = -1;
  prev_street_contrib_ = 0;
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
  UpdateHistory(my_contribution, opp_contribution, street);

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
  PrintHistory(std::vector<int>(history_.begin(), history_.end()));
  // PrintVector(std::vector<int>(history_.begin(), history_.end()));
  std::cout << std::endl;
}

// This is always called when it's my action.
void Player::UpdateHistory(int my_contrib, int opp_contrib, int street) {
  const bool did_start_new_street = (prev_street_ != street);

  assert(my_contrib >= contributions_[0]);
  assert(opp_contrib >= contributions_[1]);

  if (did_start_new_street) {
    // std::cout << "[HISTORY] Started new street: " << street << std::endl;
    // std::cout << "[HISTORY] prev_street=" << prev_street_ << std::endl;
    prev_street_ = street;

    // If we don't have a previous street, don't need to do call detection below.
    if (street > 0) {
      // Make sure the previous street has equal pot contributions.
      const int this_street_off = kMaxActionsPerStreet * GetStreet0123(street);
      const int prev_street_off = this_street_off - kMaxActionsPerStreet;

      // The amount that was put in the pot by each player during the last street.
      const int prev_street_pip = std::min(my_contrib, opp_contrib) - prev_street_contrib_;
      // printf("[HISTORY] this_street_off=%d | prev_street_off=%d | prev_street_pip=%d\n",
      //     this_street_off, prev_street_off, prev_street_pip);

      std::array<int, 2> pips = { 0, 0 };

      // We're looking for the call that ended the previous round of betting.
      for (int i = prev_street_off; i < this_street_off; ++i) {
        const int add_amt = history_.at(i);
        const int remaining_amt = (prev_street_pip - pips[i % 2]);
        if (add_amt == 0 && remaining_amt > 0) {
          // const int remaining_amt = (prev_street_contrib - pips[i % 2])
          // const int call_amt = std::abs(pips[0] - pips[1]); 
          history_.at(i) = remaining_amt;
          // printf("[HISTORY] Corrected the final call of %d at action %d\n", remaining_amt, i);
        }
        pips.at(i % 2) += history_.at(i);
        if (pips[0] == prev_street_pip && pips[1] == prev_street_pip) {
          break;
        }
      }

      // The next action idx should be the first of the new street.
      next_action_idx_ = kMaxActionsPerStreet * GetStreet0123(street);

      contributions_[0] = std::min(my_contrib, opp_contrib);
      contributions_[1] = std::min(my_contrib, opp_contrib);
      // printf("Both players starting out new street with contributions: US=%d | OPP=%d\n",
          // contributions_[0], contributions_[1]);
      prev_street_contrib_ = std::min(my_contrib, opp_contrib);
    }
  }

  // printf("[HISTORY] Updating with latest action(s) my_contrib=%d | opp_contrib=%d\n", my_contrib, opp_contrib);

  // If the opp_contrib has increased, an opponent action must have happened since our last action.
  if (opp_contrib > contributions_[1]) {
    // printf("[HISTORY] opp_contrib > contributions_[1] (%d and %d)\n", opp_contrib, contributions_[1]);
    const int add_amt = (opp_contrib - contributions_[1]);
    history_.at(next_action_idx_) = add_amt;
    contributions_[1] = opp_contrib;
    
    // If we surpass kMaxActionsPerStreet, keeping adding contributions to the LAST betting action.
    ++next_action_idx_;
    const int parity = next_action_idx_ % 2;
    const int max_idx_this_street = kMaxActionsPerStreet * (GetStreet0123(street) + 1);
    next_action_idx_ = std::min(next_action_idx_, max_idx_this_street - 2 + parity);
    // printf("Updated opp action, next_action_idx=%d\n", next_action_idx_);
  }

  // If my_contrib has increased, we must have taken an action.
  if (my_contrib > contributions_[0]) {
    // printf("[HISTORY] my_contrib > contributions_[0] (%d and %d)\n", my_contrib, contributions_[0]);
    history_.at(next_action_idx_) = (my_contrib - contributions_[0]);
    contributions_[0] = my_contrib;

    // If we surpass kMaxActionsPerStreet, keeping adding contributions to the LAST betting action.
    ++next_action_idx_;
    const int parity = next_action_idx_ % 2;
    const int max_idx_this_street = kMaxActionsPerStreet * (GetStreet0123(street) + 1);
    next_action_idx_ = std::min(next_action_idx_, max_idx_this_street - 2 + parity);
    // printf("Updated our action, next_action_idx=%d\n", next_action_idx_);
  }
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

  UpdateHistory(my_contribution, opp_contribution, street);

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

  // if (did_converge) {
  const Action action = HandleActionConverged(
      EV, round_num, street, pot_size, continue_cost, legal_actions,
      min_raise, max_raise, my_contribution, opp_contribution);

  return action;
}

Action Player::HandleActionConverged(float EV, int round_num, int street, int pot_size,
                                     int continue_cost, int legal_actions, int min_raise,
                                     int max_raise, int my_contribution, int opp_contribution) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  printf("check_is_allowed=%d | raise_is_allowed=%d\n", check_is_allowed, raise_is_allowed);

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

// Bet less agressively, but also lower the thresholds for calling so that we are more likely to
// see a showdown and gather information.
Action Player::HandleActionNotConverged(float EV, int round_num, int street, int pot_size,
                                     int continue_cost, int legal_actions, int min_raise,
                                     int max_raise, int my_contribution, int opp_contribution) {
  std::cout << "HandleActionNotConverged" << std::endl;
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  printf("check_is_allowed=%d | raise_is_allowed=%d\n", check_is_allowed, raise_is_allowed);

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.7 || !raise_is_allowed) {
      std::cout << "[WAITING] (CheckAllowed) EV < 0.7 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If hand is really good, do a pot raise.
    } else {
      std::cout << "[WAITING] (CheckAllowed) EV >= 0.7 ==> PotRaise" << std::endl;
      const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
      return RaiseAction(raise_amt);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const int equity = EV * pot_after_call;
    printf("[WAITING] Call equity | equity=%d | pot_after_call=%d | continue_cost=%d\n", equity, pot_after_call, continue_cost);

    // Not worth it to continue.
    if (equity < (1.2 * continue_cost)) {
      std::cout << "[WAITING] (CallRequired) not worth it ==> FoldAction" << std::endl;
      return FoldAction();

    // Worth it - do bet sizing.
    } else {
      if (EV >= 0.7) {
        const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
        std::cout << "[WAITING] (CallRequired) worth it, EV >= 0.8 ==> PotRaise" << std::endl;
        return raise_is_allowed ? RaiseAction(raise_amt) : CallAction();
      } else {
        std::cout << "[WAITING] (CallRequired) worth it, but EV not >= 0.8 ==> CallAction" << std::endl;
        return CallAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

}
