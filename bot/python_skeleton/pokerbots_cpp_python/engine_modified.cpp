/**
 * Encapsulates round state information for the player.
 */
#include "engine_modified.hpp"
#include <map>
#include <omp/HandEvaluator.h>

namespace pb {

omp::HandEvaluator OMP_{};

/**
 * Compares the players' hands and computes payoffs.
 */
State* RoundState::showdown()
{
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

    return new TerminalState(std::array<int, 2>({delta, -delta}), this);
}

/**
 * Returns a mask which corresponds to the active player's legal moves.
 */
int RoundState::legal_actions()
{
    int active = this->button % 2;
    int continue_cost = this->pips[1-active] - this->pips[active];
    if (continue_cost == 0)
    {
        // we can only raise the stakes if both players can afford it
        bool bets_forbidden = ((this->stacks[0] == 0) | (this->stacks[1] == 0));
        if (bets_forbidden)
        {
            return CHECK_ACTION_TYPE;
        }
        return CHECK_ACTION_TYPE | RAISE_ACTION_TYPE;
    }
    // continue_cost > 0
    // similarly, re-raising is only allowed if both players can afford it
    bool raises_forbidden = ((continue_cost == this->stacks[active]) | (this->stacks[1-active] == 0));
    if (raises_forbidden)
    {
        return FOLD_ACTION_TYPE | CALL_ACTION_TYPE;
    }
    return FOLD_ACTION_TYPE | CALL_ACTION_TYPE | RAISE_ACTION_TYPE;
}

/**
 * Returns an array of the minimum and maximum legal raises.
 */
array<int, 2> RoundState::raise_bounds()
{
    int active = this->button % 2;
    int continue_cost = this->pips[1-active] - this->pips[active];
    int max_contribution = min(this->stacks[active], this->stacks[1-active] + continue_cost);
    int min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND));
    return (array<int, 2>) { this->pips[active] + min_contribution, this->pips[active] + max_contribution };
}

/**
 * Resets the players' pips and advances the game tree to the next round of betting.
 */
State* RoundState::proceed_street()
{
    if (this->street == 5)
    {
        return this->showdown();
    }
  
    bet_history.emplace_back(std::vector<int>());

    int new_street;
    if (this->street == 0)
    {
        new_street = 3;
    }
    else
    {
        new_street = this->street + 1;
    }
    return new RoundState(1 - sb_player, new_street, (array<int, 2>) { 0, 0 }, this->stacks, this->hands, this->deck, this, bet_history, sb_player);
}

/**
 * Advances the game tree by one action performed by the active player.
 */
State* RoundState::proceed(Action action)
{
    int active = this->button % 2;
    switch (action.action_type)
    {
        case FOLD_ACTION_TYPE:
        {
            int delta;
            if (active == 0)
            {
                delta = this->stacks[0] - STARTING_STACK;
            }
            else
            {
                delta = STARTING_STACK - this->stacks[1];
            }
            return new TerminalState((array<int, 2>) { delta, -1 * delta }, this);
        }
        case CALL_ACTION_TYPE:
        {
            if (this->button == 0)  // sb calls bb
            {
                bet_history.back().emplace_back(1);
                return new RoundState(button + 1, 0, (array<int, 2>) { BIG_BLIND, BIG_BLIND },
                                      (array<int, 2>) { STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND },
                                      this->hands, this->deck, this, bet_history, sb_player);
            }
            // both players acted
            array<int, 2> new_pips = this->pips;
            array<int, 2> new_stacks = this->stacks;
            int contribution = new_pips[1-active] - new_pips[active];
            new_stacks[active] -= contribution;
            new_pips[active] += contribution;
            bet_history.back().emplace_back(contribution);
            RoundState* state = new RoundState(this->button + 1, this->street, new_pips, new_stacks,
                                               this->hands, this->deck, this, bet_history, sb_player);
            return state->proceed_street();
        }
        case CHECK_ACTION_TYPE:
        {
            if (((this->street == 0) & (this->button > 0)) | (this->button > 1))  // both players acted
            {
                bet_history.back().emplace_back(0);
                return this->proceed_street();
            }
            // let opponent act
            bet_history.back().emplace_back(0);
            return new RoundState(this->button + 1, this->street, this->pips, this->stacks, this->hands, this->deck, this, bet_history, sb_player);
        }
        default:  // RAISE_ACTION_TYPE
        {
            array<int, 2> new_pips = this->pips;
            array<int, 2> new_stacks = this->stacks;
            int contribution = action.amount - new_pips[active];
            new_stacks[active] -= contribution;
            new_pips[active] += contribution;
            bet_history.back().emplace_back(contribution);
            return new RoundState(this->button + 1, this->street, new_pips, new_stacks, this->hands, this->deck, this, bet_history, sb_player);
        }
    }
}

}
