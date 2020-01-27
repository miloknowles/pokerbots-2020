#include "cfr_trainer.hpp"
#include "engine_modified.hpp"


int main(int argc, char const *argv[]) {
  pb::cfr::Options opt;
  opt.BUCKET_FUNCTION = pb::cfr::BucketMedium;
  opt.EXPERIMENT_NAME = "MC_CFR_MEDIUM";
  opt.EXPERIMENT_PATH = "/home/milo/pokerbots-2020/cfr/" + opt.EXPERIMENT_NAME + "/";
  pb::cfr::CfrTrainer trainer(opt);
  trainer.Run();
  std::cout << "Finished running CFR" << std::endl;
  return 0;
}
