/**
 * Encapsulates round state information for the player.
 */
#include "engine_modified.hpp"
#include <map>
#include <omp/HandEvaluator.h>
#include <iostream>

namespace pb {
namespace cfr {

omp::HandEvaluator OMP_{};

/**
 * Compares the players' hands and computes payoffs.
 */
RoundState RoundState::showdown() const {
    omp::Hand h0 = omp::Hand::empty();
    const std::string h0_and_board = hands[0][0] + hands[0][1] + deck[0] + deck[1] + deck[2] + deck[3] + deck[4];
    for (int i = 0; i < 7; ++i) {
        const uint8_t code = 4 * utils::RANK_STR_TO_VAL[h0_and_board[2*i]] + utils::SUIT_STR_TO_VAL[h0_and_board[2*i+1]];
        h0 += omp::Hand(code);
    }

    omp::Hand h1 = omp::Hand::empty();
    const std::string h1_and_board = hands[1][0] + hands[1][1] + deck[0] + deck[1] + deck[2] + deck[3] + deck[4];
    for (int i = 0; i < 7; ++i) {
        const uint8_t code = 4 * utils::RANK_STR_TO_VAL[h1_and_board[2*i]] + utils::SUIT_STR_TO_VAL[h1_and_board[2*i+1]];
        h1 += omp::Hand(code);
    }

    const uint16_t score0 = OMP_.evaluate(h0);
    const uint16_t score1 = OMP_.evaluate(h1);

    int delta = 0;
    if (score0 > score1) {
        delta = STARTING_STACK - stacks[1];
    } else if (score0 < score1) {
        delta = stacks[0] - STARTING_STACK;
    } else {
        delta = (stacks[0] - stacks[1]) / 2;
    }

    return RoundState(this->button, this->street, this->pips, this->stacks, this->hands,
                    this->deck, this->bet_history, this->sb_player, true,
                    std::array<int, 2>({delta, -delta}));
}

/**
 * Returns a mask which corresponds to the active player's legal moves.
 */
int RoundState::legal_actions() const {
    int active = this->button % 2;
    int continue_cost = this->pips[1-active] - this->pips[active];
    if (continue_cost == 0) {
        // we can only raise the stakes if both players can afford it
        bool bets_forbidden = ((this->stacks[0] == 0) | (this->stacks[1] == 0));
        if (bets_forbidden) {
            return CHECK_ACTION_TYPE;
        }
        return CHECK_ACTION_TYPE | RAISE_ACTION_TYPE;
    }
    // continue_cost > 0
    // similarly, re-raising is only allowed if both players can afford it
    bool raises_forbidden = ((continue_cost == this->stacks[active]) | (this->stacks[1-active] == 0));
    if (raises_forbidden) {
        return FOLD_ACTION_TYPE | CALL_ACTION_TYPE;
    }
    return FOLD_ACTION_TYPE | CALL_ACTION_TYPE | RAISE_ACTION_TYPE;
}

/**
 * Returns an array of the minimum and maximum legal raises.
 */
array<int, 2> RoundState::raise_bounds() const {
    int active = this->button % 2;
    int continue_cost = this->pips[1-active] - this->pips[active];
    int max_contribution = min(this->stacks[active], this->stacks[1-active] + continue_cost);
    int min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND));
    return (array<int, 2>) { this->pips[active] + min_contribution, this->pips[active] + max_contribution };
}

/**
 * Resets the players' pips and advances the game tree to the next round of betting.
 */
RoundState RoundState::proceed_street() const {
    if (this->street == 5) {
        return this->showdown();
    }
    
    BetHistory new_bet_history = this->bet_history;
    new_bet_history.emplace_back(std::vector<int>());

    int new_street;
    if (this->street == 0) {
        new_street = 3;
    } else {
        new_street = this->street + 1;
    }
    return RoundState(1 - sb_player, new_street, (array<int, 2>) { 0, 0 }, this->stacks,
                    this->hands, this->deck, new_bet_history, this->sb_player, false,
                    std::array<int, 2>({-1, -1}));
}

/**
 * Advances the game tree by one action performed by the active player.
 */
RoundState RoundState::proceed(Action action) const {
    int active = this->button % 2;
    switch (action.action_type) {
        case FOLD_ACTION_TYPE: {
            int delta;
            if (active == 0) {
                delta = this->stacks[0] - STARTING_STACK;
            }
            else {
                delta = STARTING_STACK - this->stacks[1];
            }
            return RoundState(this->button, this->street, this->pips, this->stacks, this->hands,
                              this->deck, this->bet_history, this->sb_player, true,
                              (array<int, 2>) { delta, -1 * delta });
        }
        case CALL_ACTION_TYPE: {
            // SB calls BB.
            if (this->button == this->sb_player && this->street == 0) {
                BetHistory new_bet_history = this->bet_history;
                new_bet_history.back().emplace_back(1);
                return RoundState(this->button + 1, 0, (array<int, 2>) { BIG_BLIND, BIG_BLIND },
                                 (array<int, 2>) { STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND },
                                 this->hands, this->deck, new_bet_history, this->sb_player, false,
                                 std::array<int, 2>({-1, -1}));
            }
            // both players acted
            array<int, 2> new_pips = this->pips;
            array<int, 2> new_stacks = this->stacks;
            const int contribution = new_pips[1-active] - new_pips[active];
            new_stacks[active] -= contribution;
            new_pips[active] += contribution;

            BetHistory new_bet_history = bet_history;
            new_bet_history.back().emplace_back(contribution);

            return RoundState(this->button + 1, this->street, new_pips, new_stacks,
                            this->hands, this->deck, new_bet_history, this->sb_player,
                            false, std::array<int, 2>({-1, -1})).proceed_street();
            // return state.proceed_street();
        }
        case CHECK_ACTION_TYPE: {
            // if (self.street == 0 and self.button > 0) or self.button > 1:  # both players acted
            if ((this->street == 0 && this->button > sb_player) || (this->button % 2 == this->sb_player)) {
                BetHistory new_bet_history = this->bet_history;
                new_bet_history.back().emplace_back(0);
                return RoundState(this->button, this->street, this->pips, this->stacks, this->hands,
                                 this->deck, new_bet_history, this->sb_player, false, std::array<int, 2>({-1, -1})).proceed_street();
                // return this->proceed_street();
            }

            // let opponent act
            BetHistory new_bet_history = this->bet_history;
            new_bet_history.back().emplace_back(0);
            return RoundState(this->button + 1, this->street, this->pips, this->stacks, this->hands,
                            this->deck, new_bet_history, this->sb_player, false, std::array<int, 2>({-1, -1}));
        }
    
        // RAISE ACTION TYPE
        default: {
            array<int, 2> new_pips = this->pips;
            array<int, 2> new_stacks = this->stacks;
            int contribution = action.amount - new_pips[active];
            new_stacks[active] -= contribution;
            new_pips[active] += contribution;
            BetHistory new_bet_history = this->bet_history;
            new_bet_history.back().emplace_back(contribution);
            return RoundState(this->button + 1, this->street, new_pips, new_stacks, this->hands,
                              this->deck, new_bet_history, this->sb_player, false, std::array<int, 2>({-1, -1}));
        }
    }
}

}
}
