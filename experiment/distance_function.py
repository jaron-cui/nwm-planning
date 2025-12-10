# distance function
import os
# os.chdir('C:/Users/clack/Projects/nwm')

from typing import Tuple
import torch
import torch.nn as nn
from experiment.vae import Encoder

class DistancePredictor(nn.Module):
  def __init__(self, bottleneck_channels: int, hidden_dim: int, classes: int) -> None:
    super().__init__()
    self.convolutional_layers = Encoder(
      block_config_str='8x6,8d2,8t4,4x4,4d4,4t1,1x4',
      channel_config_str='8:256,4:256,1:512',
      bottleneck_channels=bottleneck_channels,
      image_channels=8
    )
    self.ffwd = nn.Sequential(
      nn.ReLU(),
      nn.Linear(bottleneck_channels, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU()
    )
    self.distance_fc = nn.Sequential(
      nn.Linear(hidden_dim, classes)
    )
    # note: not using these right now, but it's more convenient to let my old checkpoints load without issue
    self.x_displacement_fc = nn.Sequential(
      nn.Linear(hidden_dim, classes)
    )
    self.y_displacement_fc = nn.Sequential(
      nn.Linear(hidden_dim, classes)
    )

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
    convolved, _ = self.convolutional_layers(x)
    out = self.ffwd(convolved.reshape(x.shape[0], -1))
    return self.distance_fc(out)#, self.x_displacement_fc(out), self.y_displacement_fc(out)

def make_distance_predictor():
  return DistancePredictor(32, 128, 64)
