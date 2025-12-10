from typing import Callable, Sequence
import numpy as np
import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusion import create_diffusion
from misc import transform, unnormalize
from models import CDiT_models

def load_cdit(checkpoint_path: str) -> Callable[[Sequence[torch.Tensor], Sequence[np.ndarray]], torch.Tensor]:
  EXP_NAME = 'nwm_cdit_s'

  with open("config/data_config.yaml", "r") as f:
      default_config = yaml.safe_load(f)
  config = default_config

  with open(f'config/{EXP_NAME}.yaml', "r") as f:
      user_config = yaml.safe_load(f)
  config.update(user_config)
  latent_size = config['image_size'] // 8
  print(latent_size)
  print("loading model")
  import pathlib
  pathlib.PosixPath = pathlib.WindowsPath
  model = CDiT_models['CDiT-S/2'](input_size=latent_size, context_size=config['context_size'])
  ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False) 

  print(model.load_state_dict(ckp["ema"], strict=True))
  model.eval()
  device = 'cuda'
  model.to(device)
  model = torch.compile(model)

  diffusion = create_diffusion(str(250))

  @torch.no_grad()
  def cdit_forward(latents: torch.Tensor, actions: np.ndarray, progress: bool = False):
    y = torch.cat((torch.tensor(actions), torch.zeros((*actions.shape[:-1], 1))), dim=-1).to(device)
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
      z = torch.randn(1, 4, latent_size, latent_size, device=device)
      y = y.flatten(0, 1)
      model_kwargs = dict(y=y, x_cond=latents.unsqueeze(0), rel_t=torch.ones(1, device=device))      
      latents = diffusion.p_sample_loop(
      model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
      )
    return latents
  return lambda states, actions: cdit_forward(torch.stack(list(states)), np.stack(actions))
    
