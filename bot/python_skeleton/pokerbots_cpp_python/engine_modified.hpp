#pragma once

#include <array>
#include <vector>
#include <string>
#include <map>
#include "actions.hpp"

using std::array;
using std::string;
using std::max;
using std::min;

namespace pb {

const int NUM_ROUNDS     = 1000;
const int STARTING_STACK = 200;
const int BIG_BLIND      = 2;
const int SMALL_BLIND    = 1;

typedef std::vector<std::vector<int>> BetHistory;


namespace utils {

static std::map<char, uint8_t> RANK_STR_TO_VAL = {
  {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3}, {'6', 4}, {'7', 5}, {'8', 6},
  {'9', 7}, {'T', 8}, {'J', 9}, {'Q', 10}, {'K', 11}, {'A', 12}
};

static std::map<uint8_t, char> RANK_VAL_TO_STR = {
  {0, '2'}, {1, '3'}, {2, '4'}, {3, '5'}, {4, '6'}, {5, '7'}, {6, '8'},
  {7, '9'}, {8, 'T'}, {9, 'J'}, {10, 'Q'}, {11, 'K'}, {12, 'A'}
};

// Defined by OMPEval.
static std::map<char, uint8_t> SUIT_STR_TO_VAL = {
  {'s', 0}, {'h', 1}, {'c', 2}, {'d', 3}
};

static std::map<uint8_t, char> SUIT_VAL_TO_STR = {
  {0, 's'}, {1, 'h'}, {2, 'c'}, {3, 'd'}
};

}


/**
 * Stores higher state information across many rounds of poker.
 */
class GameState
{
    public:
        const int bankroll;
        const float game_clock;
        const int round_num;

        GameState(int bankroll, float game_clock, int round_num):
            bankroll(bankroll),
            game_clock(game_clock),
            round_num(round_num)
        {}
};


/**
 * The base class for the current state of one round of poker.
 */
class State
{
    virtual void foo() {}
};


/**
 * Final state of a poker round corresponding to payoffs.
 */
class TerminalState : public State
{
    public:
        const array<int, 2> deltas;
        State* const previous_state;

        TerminalState(array<int, 2> deltas, State* previous_state):
            deltas(deltas),
            previous_state(previous_state)
        {}
};


/**
 * Encodes the game tree for one round of poker.
 */
class RoundState : public State
{
    public:
        const int button;
        const int street;
        const array<int, 2> pips;
        const array<int, 2> stacks;
        const array< array<string, 2>, 2 > hands;
        const array<string, 5> deck;
        State* const previous_state;
        BetHistory bet_history;
        const int sb_player;

        RoundState(int button,
                   int street,
                   array<int, 2> pips,
                   array<int, 2> stacks,
                   array< array<string, 2>, 2 > hands,
                   array<string, 5> deck,
                   State* previous_state,
                   const BetHistory& bet_history,
                   const int sb_player):
            button(button),
            street(street),
            pips(pips),
            stacks(stacks),
            hands(hands),
            deck(deck),
            previous_state(previous_state),
            bet_history(bet_history),
            sb_player(sb_player) {}

        /**
         * Compares the players' hands and computes payoffs.
         */
        State* showdown();

        /**
         * Returns a mask which corresponds to the active player's legal moves.
         */
        int legal_actions();

        /**
         * Returns an array of the minimum and maximum legal raises.
         */
        array<int, 2> raise_bounds();

        /**
         * Resets the players' pips and advances the game tree to the next round of betting.
         */
        State* proceed_street();

        /**
         * Advances the game tree by one action performed by the active player.
         */
        State* proceed(Action action);
};

}
