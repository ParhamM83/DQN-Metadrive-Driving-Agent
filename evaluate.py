import os
import sys
import json
import argparse
import numpy as np
import torch

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.agent import DQNAgent


def evaluate_agent(
    num_episodes: int = 100,
    start_seed: int = 21,
    num_scenarios: int = 100,
    model_path: str = "models/dqn_trained.pt",
    output_path: str = "Evaluation/evaluation_report.json",
    device: str = "cpu",
):
    if not os.path.isfile(model_path):
        print(f"[!] Error: Model checkpoint not found at '{model_path}'", file=sys.stderr)
        print("    Please ensure training has completed or provide a valid --model-path.", file=sys.stderr)
        sys.exit(1)

    # Initialize environment with zero-shot unseen seed configuration
    env = create_env(start_seed=start_seed, num_scenarios=num_scenarios, render=False)

    state_size = 259
    action_size = 6
    hidden_size = 512
    agent = DQNAgent(
        state_size=state_size,
        hidden_size=hidden_size,
        action_size=action_size,
        device=device,
    )

    agent.load(model_path, map_location=device)
    agent.policy_net.eval()

    total_rewards = []
    episode_steps = []
    success_count = 0
    out_of_road_count = 0
    crash_vehicle_count = 0

    print(f"[*] Starting zero-shot evaluation on {num_episodes} episodes...")
    print(f"    Map seed range: [{start_seed} -> {start_seed + num_scenarios - 1}]")
    print("-" * 65)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)
        episode_reward = 0.0
        steps = 0

        while True:
            # Greedy action selection (epsilon=0.0)
            action_idx = agent.act(state, epsilon=0.0)
            continuous_action = discrete_to_continuous_action(action_idx)

            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated

            state = flatten_obs(next_obs)
            episode_reward += reward
            steps += 1

            if done:
                if info.get("arrive_dest", False):
                    success_count += 1
                    reason = "SUCCESS"
                elif info.get("out_of_road", False):
                    out_of_road_count += 1
                    reason = "OUT_OF_ROAD"
                else:
                    crash_vehicle_count += 1
                    reason = "CRASH_VEHICLE"
                break

        total_rewards.append(episode_reward)
        episode_steps.append(steps)
        print(
            f"Eval Episode {episode:3d}/{num_episodes:3d} | "
            f"Steps: {steps:3d} | Reward: {episode_reward:7.2f} | Result: {reason}"
        )

    # Aggregated metrics
    average_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    average_steps = float(np.mean(episode_steps))
    success_rate = (success_count / num_episodes) * 100.0
    out_of_road_rate = (out_of_road_count / num_episodes) * 100.0
    crash_vehicle_rate = (crash_vehicle_count / num_episodes) * 100.0

    print("=" * 65)
    print("                    EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Total Episodes Tested:    {num_episodes}")
    print(f"  Unseen Scenarios:         {num_scenarios} (seed start: {start_seed})")
    print(f"  Average Reward:           {average_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Average Episode Length:   {average_steps:.1f} steps")
    print(f"  Success Rate:             {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"  Out-of-Road Rate:         {out_of_road_rate:.1f}% ({out_of_road_count}/{num_episodes})")
    print(f"  Vehicle Crash Rate:       {crash_vehicle_rate:.1f}% ({crash_vehicle_count}/{num_episodes})")
    print("=" * 65)

    report_data = {
        "average_reward": round(average_reward, 2),
        "reward_std": round(std_reward, 2),
        "average_steps": round(average_steps, 1),
        "success_rate": round(success_rate, 1),
        "out_of_road_rate": round(out_of_road_rate, 1),
        "crash_vehicle_rate": round(crash_vehicle_rate, 1),
        "tested_scenarios": num_scenarios,
        "start_seed": start_seed,
        "total_episodes": num_episodes,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"[+] Evaluation metrics saved to '{output_path}'")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of trained DQN agent across unseen maps")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=21, help="Starting map seed for evaluation")
    parser.add_argument("--scenarios", type=int, default=100, help="Number of different maps to test")
    parser.add_argument("--model-path", type=str, default="models/dqn_trained.pt", help="Path to saved .pt model checkpoint")
    parser.add_argument("--output", type=str, default="Evaluation/evaluation_report.json", help="Path to save output JSON report")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation ('cpu' or 'cuda')")

    args = parser.parse_args()
    evaluate_agent(
        num_episodes=args.episodes,
        start_seed=args.seed,
        num_scenarios=args.scenarios,
        model_path=args.model_path,
        output_path=args.output,
        device=args.device,
    )