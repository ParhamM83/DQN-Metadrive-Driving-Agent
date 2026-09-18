import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

# Pre-allocated continuous control vectors [Steering, Acceleration/Brake]
# Eliminates parking/stalling with minimum forward throttle 0.1
ACTION_MAP = {
    0: np.array([-1.00, 0.1], dtype=np.float32),  # 0: Hard Left + Engine Brake (Dodging)
    1: np.array([-0.75, 0.6], dtype=np.float32),  # 1: Mild Left + Forward (High-speed cornering)
    2: np.array([ 0.00, 0.1], dtype=np.float32),  # 2: Straight + Engine Brake (Traffic slowing, no parking)
    3: np.array([ 0.00, 0.8], dtype=np.float32),  # 3: Straight + Gas (Speed / progress accumulator)
    4: np.array([ 1.00, 0.1], dtype=np.float32),  # 4: Hard Right + Engine Brake (Dodging)
    5: np.array([ 0.75, 0.6], dtype=np.float32),  # 5: Mild Right + Forward (High-speed cornering)
}
DEFAULT_ACTION = np.array([0.0, 0.0], dtype=np.float32)


def create_env(start_seed: int = 1, num_scenarios: int = 20, render: bool = False) -> MetaDriveEnv:
    """Creates and configures the MetaDrive single-agent driving environment.

    Args:
        start_seed: Seed defining the starting procedural road layout.
        num_scenarios: Total number of unique road layouts to sample from.
        render: Whether to enable 3D window rendering.
    """
    config = {
        "num_scenarios": num_scenarios,
        "start_seed": start_seed,
        "use_render": render,
        "crash_vehicle_done": True,   # Terminate immediately on vehicle collision
        "out_of_route_done": True,    # Terminate immediately if vehicle leaves road
        "crash_vehicle_penalty": 50.0,
        "out_of_road_penalty": 50.0,
        "success_reward": 200.0,
    }
    return MetaDriveEnv(config)


def flatten_obs(obs) -> np.ndarray:
    """Flattens MetaDrive observation into a 1D float32 numpy vector."""
    if isinstance(obs, dict):
        parts = [np.asarray(v, dtype=np.float32).flatten() for _, v in sorted(obs.items())]
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).flatten()


def discrete_to_continuous_action(action_idx: int) -> np.ndarray:
    """Translates discrete action index (0-5) into continuous controls [Steering, Throttle/Brake]."""
    return ACTION_MAP.get(action_idx, DEFAULT_ACTION)