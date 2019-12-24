import ray
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@ray.remote
def f(i):
  time.sleep(0.1)
  return i*i

class SimpleNetwork(nn.Module):
  def __init__(self):
    super(SimpleNetwork, self).__init__()
    self.linear = nn.Linear(10, 10)

  def forward(self, x):
    return F.relu(self.linear(x))

@ray.remote
def run_process(net, device):
  for i in range(10000):
    with torch.no_grad():
      x = torch.zeros(10)
      out = net.forward.remote(x)
      y = ray.get(out)
      # y = out.get(timeout=1.0)

  return y

@ray.remote(num_gpus=1)
class NetworkWrapper():
  def __init__(self, device):
    self._device = device
    self._net = SimpleNetwork().to(device)

  def forward(self, x):
    return self._net(x.to(self._device)).cpu() 


if __name__ == "__main__":
  ray.init()

  device = torch.device("cuda:0")
  # net = SimpleNetwork().to(device)

  # remote_net = ray.remote(NetworkWrapper)
  net_actor = NetworkWrapper.remote(device)

  # x = torch.zeros(10).to(device)
  x = torch.zeros(10)
  y = net_actor.forward.remote(x)

  out = ray.get([run_process.remote(net_actor, device) for _ in range(8)])
  print(out)
