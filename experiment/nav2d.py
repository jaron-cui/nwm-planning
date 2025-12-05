from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


from typing import List, Tuple
import numpy as np


class Topo:
  def __init__(self, seed: int, terrain_size: int, wall_height: float, noise_resolution: float, avatar_width: float) -> None:
    self.seed = seed
    self.terrain_size = terrain_size
    self.wall_height = wall_height
    self.avatar_width = avatar_width

    rng = np.random.Generator(np.random.PCG64(seed))
    self.terrain = sample_2d_perlin_noise_region(
      rng,
      np.zeros(2),
      np.full(2, terrain_size - 1),
      (terrain_size, terrain_size),
      noise_resolution
    )
    self.position = np.full(2, terrain_size / 2) + rng.random(2) - 0.5

  @staticmethod
  def random(
    terrain_size: int = 256,
    wall_height: float = 0.52,
    noise_resolution: float = 15,
    avatar_width: float = 0.5,
    avoid_wall_start: bool = True
  ) -> 'Topo':
    state = Topo(np.random.randint(0, 2**31), terrain_size, wall_height, noise_resolution, avatar_width)
    r = avatar_width / 2
    while avoid_wall_start and any([state.terrain[int((state.position[1] + yo).round()), int((state.position[0] + xo).round())] >= wall_height for xo, yo in [[0, 0], [r, r], [r, -r], [-r, r], [-r, r]]]):
      state = Topo(np.random.randint(0, 2**31), terrain_size, wall_height, noise_resolution, avatar_width)
    return state

  def act(self, action: np.ndarray):
    r = self.avatar_width / 2
    corners = self.position[None, :] + np.array([[0, 0], [r, r], [r, -r], [-r, r], [-r, r]])
    action_progression = np.stack([self._cast_ray(ray_origin=corner, action=action) for corner in corners]).min(0)
    self.position += action_progression * action

  def _cast_ray(self, ray_origin: np.ndarray, action: np.ndarray) -> np.ndarray:
    first_edges = np.ceil(ray_origin - 0.5) * (action >= 0) + np.floor(ray_origin - 0.5) * (action < 0) + 0.5  # edges are on 0.5 grid offset instead of integers
    check_blocks_for_collision = []
    for axis, counteraxis in [(0, 1), (1, 0)]:
      for edge in np.arange(first_edges[axis], first_edges[axis] + action[axis], 1 if action[axis] >= 0 else -1):
        action_progression = (edge - first_edges[axis]) / action[axis]
        counter_collision_pos = int(np.round(action_progression * action[counteraxis] + ray_origin[counteraxis]).item())
        axial_collision_pos = int((np.ceil(edge) if action[axis] >= 0 else np.floor(edge)).item())
        block_coordinates = np.zeros(2, dtype=np.int32)
        block_coordinates[axis] = axial_collision_pos
        block_coordinates[counteraxis] = counter_collision_pos
        check_blocks_for_collision.append((action_progression, axis, block_coordinates))
    check_blocks_for_collision = sorted(check_blocks_for_collision, key=lambda c: c[0])

    starting_block = ray_origin.round().astype(np.int32)
    altitude = self.terrain[starting_block[1], starting_block[0]]
    action_progressions = np.zeros(2)
    collision = np.zeros(2, dtype=np.bool)
    for action_progression, axis, block_coordinates in check_blocks_for_collision:
      next_altitude = self.terrain[block_coordinates[1], block_coordinates[0]]
      if next_altitude >= altitude and next_altitude >= self.wall_height:
        collision[axis] = True
      if not collision[axis]:
         action_progressions[axis] = action_progression
      altitude = next_altitude
    action_progressions[~collision] = 1
    return action_progressions

  def render(self, camera_size: int) -> np.ndarray:
    corner1 = self.position - camera_size / 2
    floor_corner1 = np.floor(corner1).astype(np.int32)
    x_offset, y_offset = corner1 - floor_corner1

    xc, yc = floor_corner1
    tl = padded_slice(self.terrain, [slice(yc, yc + camera_size), slice(xc, xc + camera_size)])
    tr = padded_slice(self.terrain, [slice(yc, yc + camera_size), slice(xc + 1, xc + camera_size + 1)])
    bl = padded_slice(self.terrain, [slice(yc + 1, yc + camera_size + 1), slice(xc, xc + camera_size)])
    br = padded_slice(self.terrain, [slice(yc + 1, yc + camera_size + 1), slice(xc + 1, xc + camera_size + 1)])
    tl, tr, bl, br = [self._augment_altitude_render(region) for region in [tl, tr, bl, br]]
    antialiased = (
      tl * (1 - x_offset) * (1 - y_offset)
      + tr * x_offset * (1 - y_offset)
      + bl * (1 - x_offset) * y_offset
      + br * x_offset * y_offset
    )
    centerpoint_mask = np.zeros_like(antialiased[:, :, 0], dtype=np.bool)
    centerpoint_mask[camera_size // 2, camera_size // 2] = True
    return np.stack((antialiased[:, :, 0] * ~centerpoint_mask, antialiased[:, :, 1], antialiased[:, :, 2] * ~centerpoint_mask), axis=2)
  
  def _augment_altitude_render(self, region) -> np.ndarray:
     return np.stack((region, region * (region < self.wall_height), region * (region < self.wall_height)), axis=2)


def sample_2d_perlin_noise_region(rng: np.random.Generator, corner1: np.ndarray, corner2: np.ndarray, pixel_resolution: Tuple[int, int], noise_resolution: float) -> np.ndarray:
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    lerp = lambda a, b, t: a + t * (b - a)
    def gradient(h, x, y):
        g = h & 3
        return np.where(g < 2, x, -x) + np.where(g & 1 == 0, y, -y)
    
    x_coords = np.linspace(corner1[0], corner2[0], pixel_resolution[0])
    y_coords = np.linspace(corner1[1], corner2[1], pixel_resolution[1])
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    X_grid, Y_grid = X / noise_resolution, Y / noise_resolution
    X0, Y0 = np.floor(X_grid).astype(int), np.floor(Y_grid).astype(int)
    X1, Y1 = X0 + 1, Y0 + 1
    fx, fy = X_grid - X0, Y_grid - Y0
    u, v = fade(fx), fade(fy)
    
    p = np.arange(512, dtype=int)
    rng.shuffle(p[:256])
    p[256:] = p[:256]
    hash_coord = lambda x, y: p[p[x & 255] + (y & 255)]
    
    h00, h10, h01, h11 = hash_coord(X0, Y0), hash_coord(X1, Y0), hash_coord(X0, Y1), hash_coord(X1, Y1)
    g00, g10, g01, g11 = gradient(h00, fx, fy), gradient(h10, fx - 1, fy), gradient(h01, fx, fy - 1), gradient(h11, fx - 1, fy - 1)
    result = lerp(lerp(g00, g10, u), lerp(g01, g11, u), v)
    
    return np.clip((result + 1) / 2, 0, 1)


def padded_slice(array: np.ndarray, slices: List[slice], pad_value: float = 0) -> np.ndarray:
    """
    Extract a region from a 2D array with padding for out-of-bounds indices.
    
    Parameters:
    -----------
    array : np.ndarray
        Input 2D array
    slices : List[slice]
        List of slice objects defining the region to extract
    pad_value : float
        Value to use for padding out-of-bounds regions (default: 0)
    
    Returns:
    --------
    np.ndarray
        Extracted region with padding where needed
    
    Examples:
    ---------
    >>> arr = np.arange(9).reshape(3, 3)
    >>> padded_slice(arr, [slice(-1, 2), slice(-1, 2)])
    array([[0., 0., 0.],
           [0., 0., 1.],
           [0., 3., 4.]])
    """
    # Calculate output shape from slices
    out_shape = []
    for s in slices:
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else array.shape[len(out_shape)]
        step = s.step if s.step is not None else 1
        out_shape.append((stop - start + step - 1) // step)
    
    # Create output array filled with pad_value
    out = np.full(out_shape, pad_value, dtype=array.dtype)
    
    # Calculate the valid region in both input and output arrays
    src_slices = []
    dst_slices = []
    
    for i, s in enumerate(slices):
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else array.shape[i]
        step = s.step if s.step is not None else 1
        
        # Clamp source indices to array bounds
        src_start = max(0, start)
        src_stop = min(array.shape[i], stop)
        
        # Calculate corresponding destination indices
        dst_start = (src_start - start + step - 1) // step if step > 0 else 0
        dst_stop = dst_start + ((src_stop - src_start + step - 1) // step)
        
        src_slices.append(slice(src_start, src_stop, step))
        dst_slices.append(slice(dst_start, dst_stop))
    
    # Copy valid region
    out[tuple(dst_slices)] = array[tuple(src_slices)]
    
    return out


class Ortho:
  @dataclass
  class Barrier:
    axis: int
    position: float
    bounds: Tuple[float, float]

  def __init__(self, position: np.ndarray, barriers: List['Ortho.Barrier'], background_seed: int) -> None:
    self.position = position
    self.barrier_width = 0.003
    self.barriers = barriers
    self.barrier_endcaps = sum([[
      Ortho.Barrier(
        0 if barrier.axis == 1 else 1,
        bound + offset,
        (barrier.position - self.barrier_width, barrier.position + self.barrier_width)
      ) for bound, offset in zip(barrier.bounds, [self.barrier_width, -self.barrier_width])
    ] for barrier in barriers], [])
    self.background_seed = background_seed

  def copy(self) -> 'Ortho':
    new_state = Ortho(np.zeros(()), [], 0)
    new_state.position = self.position.copy()
    new_state.barrier_width = self.barrier_width

    # assumes barriers and barrier_endcaps are immutable
    new_state.barriers = self.barriers
    new_state.barrier_endcaps = self.barrier_endcaps

    new_state.background_seed = self.background_seed
    return new_state

  @staticmethod
  def random(
    barriers_per_dim_mean: float = 80.0,
    barriers_per_dim_std: float = 5.0,
    barrier_length_mean: float = 0.1,
    barrier_length_std: float = 0.1
  ) -> 'Ortho':
    # generate random barriers
    dims = 2
    barriers = []
    for axis, barrier_count in enumerate(
      np.random.normal(
        barriers_per_dim_mean, barriers_per_dim_std, dims
      ).round().clip(min=0).astype(np.uint32)
    ):
      barrier_positions = np.random.rand(barrier_count)
      barrier_centers = np.random.rand(barrier_count)
      barrier_lengths = np.random.normal(barrier_length_mean, barrier_length_std, barrier_count)
      for position, center, length in zip(barrier_positions, barrier_centers, barrier_lengths):
        if length <= 0:
          continue
        barriers.append(
          Ortho.Barrier(
            axis, position, (center - length / 2, center + length / 2),
          )
        )

    return Ortho(np.full(dims, 0.5), barriers, np.random.randint(0, 2**31))
  
  def act(self, action: np.ndarray, scale: float = 1.0):
    dims = 2

    effective_action = action * scale
    for axis in range(dims):
      if abs(effective_action[axis]) < 1e-8:
        continue
      alternate_axes = list(range(dims))
      alternate_axes.remove(axis)

      for barrier in self.barriers + self.barrier_endcaps:
        if barrier.axis != axis:
          continue
        start, stop = barrier.bounds
        for barrier_border in [barrier.position - self.barrier_width, barrier.position + self.barrier_width]:
          if effective_action[axis] == 0:
            continue
          distance_to_border = barrier_border - self.position[axis]
          steps_until_contact = distance_to_border / effective_action[axis]
          if steps_until_contact < 0:
            continue
          counteraxial_contact_position = self.position[alternate_axes] + steps_until_contact * effective_action[alternate_axes]
          if start < counteraxial_contact_position < stop:
            effective_action[axis] *= 1 if steps_until_contact == 0 and (effective_action[axis] > 0) != (barrier.position - self.position[axis] > 0) else min(steps_until_contact, 1)

        # if barrier.axis == axis and start <= self.position[alternate_axes] < stop:
        #   if axial_position <= barrier.position - self.barrier_width <= axial_position + offset:
        #     offset = (barrier.position - self.barrier_width) - axial_position
        #   elif axial_position >= barrier.position + self.barrier_width >= axial_position + offset:
        #     offset = (barrier.position + self.barrier_width) - axial_position
      # effective_action[axis] = offset
    self.position += effective_action

  def render(self, resolution: int, egocentric: bool = True, egocentric_camera_size: float = 0.2, point_radius: float = 0.05) -> np.ndarray:
    dims = 2
    def to_pixel_coords(array: np.ndarray, axis: int | None = None, clip: bool = False):
      if egocentric:
        pixel_position = (
          (array - self.position if axis is None else array - self.position[axis])
          / egocentric_camera_size
          + 0.5
        ) * resolution
      else:
        pixel_position = array * resolution
      return pixel_position.clip(0, resolution - 1) if clip else pixel_position
      
    canvas = np.zeros((resolution, resolution, 3))
    if egocentric:
      corner1, corner2 = np.full(2, self.position - egocentric_camera_size / 2), np.full(2, self.position + egocentric_camera_size / 2)
    else:
      corner1, corner2 = np.zeros(2), np.ones(2)
    rng = np.random.Generator(np.random.PCG64(self.background_seed))
    canvas[:, :, 2] = sample_2d_perlin_noise_region(rng, corner1, corner2, (resolution, resolution), 0.1)
    # TODO better
    x, y = to_pixel_coords(self.position).round().astype(np.int32)
    canvas[x, y, 0] = 1.0

    for barrier in self.barriers:
      counter_axis = 0 if barrier.axis == 1 else 1
      position = to_pixel_coords(np.array([barrier.position]), axis=barrier.axis).round().astype(np.int32)
      bounds = to_pixel_coords(np.array(barrier.bounds), axis=counter_axis, clip=True).round().astype(np.int32)
      pixel_coordinates = np.concatenate((position, bounds))
      if ((pixel_coordinates < 0) | (pixel_coordinates >= resolution)).any():
        continue
      indices: List[int | slice] = [0] * dims
      indices[barrier.axis] = position.item()
      indices[counter_axis] = slice(bounds[0].item(), bounds[1].item())
      canvas[indices[0], indices[1], 1] = 1.0
    # image = state.render(self.resolution, egocentric=True, egocentric_camera_size=0.2).transpose(2, 0, 1)
    canvas = canvas[:, :, 1:2].repeat(3, 2)

    return canvas


Nav2dState = Ortho | Topo


# GENERATE DATA

import os
from pathlib import Path
from typing import Callable

import cv2
import pickle
from tqdm import tqdm

def generate_dataset(dataset_dir: Path | str, trajectory_count: int, trajectory_length: int, create_policy: Callable[[Ortho], Callable[[Ortho], np.ndarray]]):
  dataset_dir = Path(dataset_dir)
  if not dataset_dir.parent.exists():
    raise FileNotFoundError(f'`{dataset_dir.parent}` does not exist.')
  os.mkdir(dataset_dir)

  for trajectory_index in tqdm(range(trajectory_count), desc='Generating trajectories'):
    trajectory_dir = dataset_dir / f'trajectory{trajectory_index}'
    os.mkdir(trajectory_dir)

    state = Ortho.random()
    policy = create_policy(state)
    position = np.zeros((trajectory_length, 2))
    for step_index in range(trajectory_length):
      image = cv2.cvtColor((state.render(resolution=64, egocentric=True, egocentric_camera_size=0.2) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
      cv2.imwrite(str(trajectory_dir / f'{step_index}.jpg'), image)
      position[step_index] = state.position
      action = policy(state)
      state.act(action, scale=1.0)
    with open(trajectory_dir / 'traj_data.pkl', 'wb') as f:
      pickle.dump({'position': position, 'yaw': np.zeros(trajectory_length)}, f)


if __name__ == '__main__':
  class WindingWalkPolicy:
    def __init__(self, step_size: float = 0.01, max_wind_per_step: float = 2.0) -> None:
      self.step_size = step_size
      self.max_angular_deviation = max_wind_per_step
      self.angle = np.random.rand() * 2 * np.pi

    def __call__(self, _) -> np.ndarray:
      self.angle += 2 * (np.random.rand() - 0.5) * self.max_angular_deviation
      return self.step_size * np.array([np.cos(self.angle), np.sin(self.angle)])

  generate_dataset('/scratch/jjc9560/datasets/nav2d/dataset1', trajectory_count=1000, trajectory_length=100, create_policy=lambda _: WindingWalkPolicy())
