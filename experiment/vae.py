# adapted from https://github.com/kpandey008/DiffuseVAE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import transform, unnormalize


def parse_layer_string(s):
  layers = []
  for ss in s.split(","):
      if "x" in ss:
          # Denotes a block repetition operation
          res, num = ss.split("x")
          count = int(num)
          layers += [(int(res), None) for _ in range(count)]
      elif "u" in ss:
          # Denotes a resolution upsampling operation
          res, mixin = [int(a) for a in ss.split("u")]
          layers.append((res, mixin))
      elif "d" in ss:
          # Denotes a resolution downsampling operation
          res, down_rate = [int(a) for a in ss.split("d")]
          layers.append((res, down_rate))
      elif "t" in ss:
          # Denotes a resolution transition operation
          res1, res2 = [int(a) for a in ss.split("t")]
          layers.append(((res1, res2), None))
      else:
          res = int(ss)
          layers.append((res, None))
  return layers


def parse_channel_string(s):
  channel_config = {}
  for ss in s.split(","):
      res, in_channels = ss.split(":")
      channel_config[int(res)] = int(in_channels)
  return channel_config


def get_conv(
  in_dim,
  out_dim,
  kernel_size,
  stride,
  padding,
  zero_bias=True,
  zero_weights=False,
  groups=1,
):
  c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
  if zero_bias:
      c.bias.data *= 0.0
  if zero_weights:
      c.weight.data *= 0.0
  return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
  return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
  return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
  def __init__(
      self,
      in_width,
      middle_width,
      out_width,
      down_rate=None,
      residual=False,
      use_3x3=True,
      zero_last=False,
  ):
    super().__init__()
    self.down_rate = down_rate
    self.residual = residual
    self.ln1 = nn.LayerNorm(in_width)
    self.ln2 = nn.LayerNorm(middle_width)
    self.ln3 = nn.LayerNorm(middle_width)
    self.ln4 = nn.LayerNorm(middle_width)
    self.c1 = get_1x1(in_width, middle_width)
    self.c2 = (
        get_3x3(middle_width, middle_width)
        if use_3x3
        else get_1x1(middle_width, middle_width)
    )
    self.c3 = (
        get_3x3(middle_width, middle_width)
        if use_3x3
        else get_1x1(middle_width, middle_width)
    )
    self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

  def forward(self, x):
      # xhat = self.c1(F.gelu(x))
      # xhat = self.c2(F.gelu(xhat))
      # xhat = self.c3(F.gelu(xhat))
      # xhat = self.c4(F.gelu(xhat))
      xhat = self.c1(F.gelu(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
      xhat = self.c2(F.gelu(self.ln2(xhat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
      xhat = self.c3(F.gelu(self.ln3(xhat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
      xhat = self.c4(F.gelu(self.ln4(xhat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
      out = x + xhat if self.residual else xhat
      if self.down_rate is not None:
          out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
      return out


class Encoder(nn.Module):
  def __init__(self, block_config_str, channel_config_str, bottleneck_channels: int, image_channels: int):
      super().__init__()
      block_config = parse_layer_string(block_config_str)
      channel_config = parse_channel_string(channel_config_str)

      self.in_conv = nn.Conv2d(image_channels, list(channel_config.values())[0], 3, stride=1, padding=1, bias=False)

      blocks = []
      for _, (res, down_rate) in enumerate(block_config):
          if isinstance(res, tuple):
              # Denotes transition to another resolution
              res1, res2 = res
              blocks.append(
                  nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
              )
              continue
          in_channel = channel_config[res]
          use_3x3 = res > 1
          blocks.append(
              ResBlock(
                  in_channel,
                  int(0.5 * in_channel),
                  in_channel,
                  down_rate=down_rate,
                  residual=True,
                  use_3x3=use_3x3,
              )
          )
      # TODO: If the training is unstable try using scaling the weights
      self.block_mod = nn.Sequential(*blocks)
      # Latents
      out_channels = list(channel_config.values())[-1]  # chennel_config[1]
      self.mu = nn.Conv2d(out_channels, bottleneck_channels, 1, bias=False)
      self.logvar = nn.Conv2d(out_channels, bottleneck_channels, 1, bias=False)

  def forward(self, input):
      x = self.in_conv(input)
      x = self.block_mod(x)
      return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
  def __init__(self, input_res, block_config_str, channel_config_str, bottleneck_channels: int, image_channels: int):
      super().__init__()
      block_config = parse_layer_string(block_config_str)
      channel_config = parse_channel_string(channel_config_str)
      blocks = []
      blocks.append(nn.Conv2d(bottleneck_channels, list(channel_config.values())[0], 1, bias=False))
      for _, (res, up_rate) in enumerate(block_config):
          if isinstance(res, tuple):
              # Denotes transition to another resolution
              res1, res2 = res
              blocks.append(
                  nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
              )
              continue

          if up_rate is not None:
              blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
              continue

          in_channel = channel_config[res]
          use_3x3 = res > 1
          blocks.append(
              ResBlock(
                  in_channel,
                  int(0.5 * in_channel),
                  in_channel,
                  down_rate=None,
                  residual=True,
                  use_3x3=use_3x3,
              )
          )
      # TODO: If the training is unstable try using scaling the weights
      self.block_mod = nn.Sequential(*blocks)
      self.last_conv = nn.Conv2d(channel_config[input_res], image_channels, 3, stride=1, padding=1)

  def forward(self, input):
      x = self.block_mod(input)
      x = self.last_conv(x)
      return torch.tanh(x)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(nn.Module):
  def __init__(
      self,
      input_res,
      enc_block_str,
      dec_block_str,
      enc_channel_str,
      dec_channel_str,
      bottleneck_channels: int,
      image_channels: int
  ):
    super().__init__()
    self.input_res = input_res
    self.enc_block_str = enc_block_str
    self.dec_block_str = dec_block_str
    self.enc_channel_str = enc_channel_str
    self.dec_channel_str = dec_channel_str

    # Encoder architecture
    self.enc = Encoder(self.enc_block_str, self.enc_channel_str, bottleneck_channels, image_channels)

    # Decoder Architecture
    self.dec = Decoder(self.input_res, self.dec_block_str, self.dec_channel_str, bottleneck_channels, image_channels)
    self.latent_scaling_factor = nn.Buffer(torch.ones(1))

  def encode(self, x):
    z = self.reparameterize(*self.enc(x))
    return z.mul_(self.latent_scaling_factor)

  def decode(self, z):
    return self.dec(z / self.latent_scaling_factor)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def compute_kl(self, mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  def forward_recons(self, x):
    # For generating reconstructions during inference
    mu, logvar = self.enc(x)
    z = self.reparameterize(mu, logvar)
    decoder_out = self.dec(z)
    return decoder_out

  def training_step(self, batch):
    x = batch

    # Encoder
    mu, logvar = self.enc(x)

    # Reparameterization Trick
    z = self.reparameterize(mu, logvar)

    # Decoder
    decoder_out = self.dec(z)

    # Compute loss
    l1_loss = F.l1_loss(decoder_out, x, reduction="sum")
    recons_loss = l1_loss
    kl_loss = self.compute_kl(mu, logvar)
    return recons_loss, kl_loss

def make_vae():
  # encoder: 3x64x64 image -> 4x8x8 latent
  enc_block_config_str = '64x4,64d2,64t32,32x4,32d2,32t16,16x4,16d2,16t8,8x3'
  enc_channel_config_str = '64:128,32:256,16:512,8:512'
  # encoder: 4x8x8 latent -> 3x64x64 reconstruction
  dec_block_config_str = '8x3,8u2,8t16,16x4,16u2,16t32,32x4,32u2,32t64,64x4'
  dec_channel_config_str = '8:512,16:512,32:256,64:128'
  return VAE(
    64,
    enc_block_config_str,
    dec_block_config_str,
    enc_channel_config_str,
    dec_channel_config_str,
    bottleneck_channels=4,
    image_channels=3
  )

@torch.no_grad()
def encode_image(vae, image: np.ndarray) -> torch.Tensor:
  x_cond_pixels = transform(torch.from_numpy(image.copy()).permute(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
  return vae.encode(x_cond_pixels)[0]

@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> np.ndarray:
    return unnormalize(vae.decode(latents)).detach().cpu().permute(0, 2, 3, 1).to(dtype=torch.float32).numpy().astype(np.float32)
