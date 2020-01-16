#include <iostream>

#include "player.hpp"

namespace pb {

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
  //int round_num = game_state->round_num;  // the round number from 1 to NUM_ROUNDS
  //std::array<std::string, 2> my_cards = round_state->hands[active];  // your cards
  //bool big_blind = (bool) active;  // true if you are the big blind
  std::cout << "CLOCK: " << game_clock << std::endl;
  street_ev_.clear();
  street_num_raises_.clear();
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
  if (round_num == 999) {
    pf_.Profile();
  }

  const bool did_see_showdown = lose_hand.size() == 4 && win_hand.size() == 4;
  if (did_see_showdown) {
    ++num_showdowns_seen_;

    const bool did_converge = pf_.Unique() == 1 && num_showdowns_seen_ > num_showdowns_converge_;

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
    std::cout << "Updated with showdown result" << std::endl;
    std::cout << "Particle filter nonzero = " << pf_.Nonzero() << std::endl;
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

  // Check fold if no particles left.
  if (pf_.Nonzero() <= 0) {
    std::cout << "[FAILURE] Particle filter empty, check-folding" << std::endl;
    return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
  }

  std::cout << "Particle filter unique = " << pf_.Unique() << std::endl;
  const bool did_converge = (num_showdowns_seen_ > num_showdowns_converge_);

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
  
  printf("\n==> ACTION round=%d street=%d ev=%f\n", round_num, street, EV);

  // Check fold if no particles left.
  if (pf_.Nonzero() <= 0) {
    std::cout << "Particle filter empty, check-folding" << std::endl;
    return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
  }

  const int pot_size = my_contribution + opp_contribution;
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.6 || !raise_is_allowed) {
      std::cout << "(CheckAllowed) EV < 0.6 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If EV is pretty good (60-80%), do a pot raise.
    } else if (EV <= 0.8) {
      const int min_raise = round_state->raise_bounds()[0];
      const int max_raise = round_state->raise_bounds()[1];
      const int raise_amt = std::min(std::max(pot_size, min_raise), max_raise);
      std::cout << "(CheckAllowed) EV <= 0.8 ==> PotRaise" << std::endl;
      return RaiseAction(raise_amt);

    // If EV above 0.8, do two pot raise.
    } else {
      const int min_raise = round_state->raise_bounds()[0];
      const int max_raise = round_state->raise_bounds()[1];
      const int raise_amt = std::min(std::max(2*pot_size, min_raise), max_raise);
      std::cout << "(CheckAllowed) EV above 0.8 ==> TwoPotRaise" << std::endl;
      return RaiseAction(max_raise);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const int equity = EV * pot_after_call;

    std::cout << "pot_after_call=" << pot_after_call << " equity=" << equity << std::endl;

    // Not worth it to continue.
    if (equity < continue_cost) {
      std::cout << "(CallRequired) not worth it ==> FoldAction" << std::endl;
      return FoldAction();

    // Worth it - do bet sizing.
    } else {
      if (EV >= 0.7) {
        const int min_raise = round_state->raise_bounds()[0];
        const int max_raise = round_state->raise_bounds()[1];
        const int raise_amt = std::min(std::max(pot_size, min_raise), max_raise);
        std::cout << "(CallRequired) worth it, EV >= 0.7 ==> PotRaise" << std::endl;
        return RaiseAction(max_raise);
      } else if (EV >= 0.9) {
        const int min_raise = round_state->raise_bounds()[0];
        const int max_raise = round_state->raise_bounds()[1];
        const int raise_amt = std::min(std::max(2*pot_size, min_raise), max_raise);
        std::cout << "(CallRequired) worth it, EV >= 0.9 ==> TwoPotRaise" << std::endl;
        return RaiseAction(max_raise);
      } else {
        std::cout << "(CallRequired) worth it, but EV not >= 0.7 ==> CallAction" << std::endl;
        return CallAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case" << std::endl;
  if (CHECK_ACTION_TYPE & legal_actions) {
    return CheckAction();
  }
  return CallAction();
}

}
