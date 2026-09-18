import os
import argparse
from collections import deque
import numpy as np
import torch

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent


def train_agent(
    num_episodes: int = 10000,
    start_seed: int = 1,
    num_scenarios: int = 20,
    batch_size: int = 256,
    hidden_size: int = 512,
    gamma: float = 0.995,
    lr: float = 3e-4,
    sync_target_freq: int = 40,
    save_dir: str = "models",
):
    # 0. Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Setup Environment
    env = create_env(start_seed=start_seed, num_scenarios=num_scenarios, render=False)

    # 2. Exploration parameters
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.05

    # 3. Initialize Agent and Replay Buffer
    state_size = 259
    action_size = 6
    memory = ReplayBuffer(capacity=100000)
    agent = DQNAgent(
        state_size=state_size,
        hidden_size=hidden_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        device=device,
    )

    # 4. Tracking arrays for reports & moving window stats
    rewards_history = []
    steps_history = []
    crash_history = []  # 0 = success, 1 = out_of_road, 2 = crash_vehicle
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "dqn_trained.pt")

    print(f"[*] Training started: {num_episodes} episodes on {num_scenarios} scenarios (seed={start_seed})")

    try:
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            state = flatten_obs(obs)

            total_reward = 0.0
            steps = 0

            while True:
                # Step A: Select action via epsilon-greedy
                action_idx = agent.act(state, epsilon)

                # Step B: Convert to continuous MetaDrive controls
                continuous_action = discrete_to_continuous_action(action_idx)

                # Step C: Step simulation
                next_obs, reward, terminated, truncated, info = env.step(continuous_action)
                done = terminated or truncated
                next_state = flatten_obs(next_obs)

                # Step D: Store transition in replay buffer
                memory.add(state, action_idx, reward, next_state, done)

                # Step E: Sample & learn once buffer has enough transitions
                if len(memory) > batch_size:
                    batch = memory.sample(batch_size)
                    agent.learn(batch)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # Categorize episode outcome
            if info.get("arrive_dest", False):
                reason = "success"
                crash_code = 0
            elif info.get("out_of_road", False):
                reason = "out_of_road"
                crash_code = 1
            else:
                reason = "crash_vehicle"
                crash_code = 2

            rewards_history.append(total_reward)
            steps_history.append(steps)
            crash_history.append(crash_code)
            recent_rewards.append(total_reward)
            recent_successes.append(1 if crash_code == 0 else 0)

            # Sync target network
            if episode % sync_target_freq == 0:
                agent.update_target_network()

            # Decay exploration rate
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            # Logging
            if episode % 10 == 0 or episode == 1 or episode == num_episodes:
                avg_100_rew = np.mean(recent_rewards)
                avg_100_succ = np.mean(recent_successes) * 100
                print(
                    f"Episode {episode:5d}/{num_episodes} | Steps: {steps:3d} | "
                    f"Reward: {total_reward:6.2f} | 100-MA Rew: {avg_100_rew:6.2f} | "
                    f"100-MA Succ: {avg_100_succ:4.1f}% | Eps: {epsilon:.3f} | Outcome: {reason}"
                )

            # Periodic checkpointing every 1,000 episodes
            if episode % 1000 == 0:
                agent.save(checkpoint_path)
                np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(rewards_history))
                np.save(os.path.join(save_dir, "steps_history.npy"), np.array(steps_history))
                np.save(os.path.join(save_dir, "crash_history.npy"), np.array(crash_history))
                print(f"[+] Checkpoint saved at episode {episode}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving current progress...")

    finally:
        # Final persistence
        agent.save(checkpoint_path)
        np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(rewards_history))
        np.save(os.path.join(save_dir, "steps_history.npy"), np.array(steps_history))
        np.save(os.path.join(save_dir, "crash_history.npy"), np.array(crash_history))
        print(f"[+] Final model and histories saved to '{save_dir}/'")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent on MetaDrive")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=1, help="Starting map seed")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of unique maps to train on")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for learning")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden units per layer")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--sync-freq", type=int, default=40, help="Episodes between target network syncs")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model & histories")
    args = parser.parse_args()

    train_agent(
        num_episodes=args.episodes,
        start_seed=args.seed,
        num_scenarios=args.scenarios,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        lr=args.lr,
        sync_target_freq=args.sync_freq,
        save_dir=args.save_dir,
    )