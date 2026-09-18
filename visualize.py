import os
import sys
import time
import argparse
import torch

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.agent import DQNAgent


def watch_agent_drive(
    view_mode: str = "3D",
    num_episodes: int = 5,
    start_seed: int = 21,
    num_scenarios: int = 100,
    model_path: str = "models/dqn_trained.pt",
):
    if not os.path.isfile(model_path):
        print(f"[!] Error: Model checkpoint not found at '{model_path}'", file=sys.stderr)
        print("    Please ensure training has completed or provide a valid --model-path.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Initializing MetaDrive environment (render={view_mode == '3D'})...")
    env = create_env(
        start_seed=start_seed,
        num_scenarios=num_scenarios,
        render=(view_mode == "3D"),
    )

    state_size = 259
    action_size = 6
    hidden_size = 512
    agent = DQNAgent(state_size=state_size, hidden_size=hidden_size, action_size=action_size, device="cpu")
    agent.load(model_path, map_location="cpu")
    agent.policy_net.eval()

    print(f"[*] Model loaded successfully from '{model_path}'")
    print(f"[*] Visualizing {num_episodes} episodes in {view_mode} mode (seed={start_seed}, scenarios={num_scenarios})...")

    try:
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            print(f"\n[Episode {episode}/{num_episodes}] Map initialized. Agent driving...")

            total_reward = 0.0
            steps = 0

            for step in range(2000):
                state = flatten_obs(obs)
                action_idx = agent.act(state, epsilon=0.0)
                continuous_action = discrete_to_continuous_action(action_idx)

                obs, reward, terminated, truncated, info = env.step(continuous_action)
                total_reward += reward
                steps += 1

                if view_mode == "top_down":
                    env.render(mode="top_down")
                    time.sleep(0.02)

                if terminated or truncated:
                    if info.get("arrive_dest", False):
                        result = "SUCCESS (Arrived at destination)"
                    elif info.get("out_of_road", False):
                        result = "OUT OF ROAD"
                    else:
                        result = "VEHICLE CRASH"

                    print(f" -> Finished in {steps} steps | Reward: {total_reward:.2f} | Result: {result}")
                    time.sleep(1.0)
                    break

    except KeyboardInterrupt:
        print("\n[!] Visualization closed by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained DQN driving agent in MetaDrive")
    parser.add_argument("--mode", type=str, default="3D", choices=["3D", "top_down"], help="Visualization mode")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=21, help="Starting map seed for testing")
    parser.add_argument("--scenarios", type=int, default=100, help="Number of unique maps to cycle through")
    parser.add_argument("--model-path", type=str, default="models/dqn_trained.pt", help="Path to trained model checkpoint")

    args = parser.parse_args()
    watch_agent_drive(
        view_mode=args.mode,
        num_episodes=args.episodes,
        start_seed=args.seed,
        num_scenarios=args.scenarios,
        model_path=args.model_path,
    )