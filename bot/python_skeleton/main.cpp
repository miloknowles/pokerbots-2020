#include "./cpp_skeleton/runner.hpp"
#include "player.hpp"

int main(int argc, char* argv[]) {
  pb::Player player;
  vector<string> args = parse_args(argc, argv);
  run_bot(&player, args);
  return 0;
}
