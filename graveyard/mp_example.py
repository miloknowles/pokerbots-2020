import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class SimpleNetwork(nn.Module):
  def __init__(self):
    super(SimpleNetwork, self).__init__()
    self.linear = nn.Linear(10, 10)

  def forward(self, x):
    return F.relu(self.linear(x))


def run_process(pid, net, device):
  for i in range(1000):
    with torch.no_grad():
      x = torch.zeros(10).to(device)
      out = net(x)

def spawn_with_cuda(nproc=2):
  net = SimpleNetwork()
  net.cuda()
  net.share_memory()

  t0 = time.time()
  mp.spawn(run_process, args=(net, torch.device("cuda:0")), nprocs=nproc, join=True, daemon=False)
  elapsed = time.time() - t0
  print("Ran in {} sec".format(elapsed))

def spawn_with_cpu(nproc=2):
  net = SimpleNetwork()
  net.share_memory()

  t0 = time.time()
  mp.spawn(run_process, args=(net, torch.device("cpu")), nprocs=nproc, join=True, daemon=False)
  elapsed = time.time() - t0
  print("Ran in {} sec".format(elapsed))


if __name__ == "__main__":
  spawn_with_cpu()
  # spawn_with_cuda()
