import os
import json
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

from environment import GeneralizedOvercooked

def plot_training_results(stats, window_size=50):
    rewards = stats['episode_rewards']
    layouts = stats['episode_layouts']
    
    if not rewards:
        print("No episodes finished during training. Cannot plot results.")
        return

    # --- Overall Reward ---
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 
             label=f'Moving Average (window={window_size})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress (Overall)')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_rewards_overall.png")
    plt.show()

    # --- Reward by Layout ---
    rewards_by_layout = defaultdict(list)
    for r, layout in zip(rewards, layouts):
        rewards_by_layout[layout].append(r)

    plt.figure(figsize=(10, 5))
    for layout, r_list in rewards_by_layout.items():
        if len(r_list) >= window_size:
            moving_avg = np.convolve(r_list, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size-1, len(r_list)), moving_avg, label=f'{layout} (avg)')
        plt.plot(r_list, alpha=0.3, label=f'{layout} (raw)')

    plt.xlabel('Episode (per layout)')
    plt.ylabel('Total Reward')
    plt.title('Training Progress by Layout')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_rewards_by_layout.png")
    plt.show()

def evaluate_agent_on_layouts(layouts, policy_path, num_episodes=5, deterministic=True,
                              render_first_episode=False, video_dir=None):    
    try:
        policy_net = tf.keras.models.load_model(policy_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    rewards_by_layout = {}
    
    for layout in layouts:
        env = GeneralizedOvercooked(layouts=[layout])
        episode_rewards = []

        for episode in range(num_episodes):
            obs_dict = env.reset()
            done = False
            total_reward = 0
            frames = []

            while not done:
                obs_batch = np.stack(obs_dict["both_agent_obs"])
                action_probs = policy_net(obs_batch, training=False).numpy()

                if deterministic:
                    actions = np.argmax(action_probs, axis=1)
                else:
                    actions = [np.random.choice(len(p), p=p) for p in action_probs]

                next_obs_dict, rewards, done, info = env.step(actions)
                total_reward += rewards
                obs_dict = next_obs_dict

                if render_first_episode or video_dir:
                    frame = env.render()
                    frames.append(frame)

            episode_rewards.append(total_reward)

            if video_dir and episode == 0:
                video_path = f"{video_dir}/{layout}_eval.mp4"
                imageio.mimsave(video_path, frames, fps=10)
                print(f"Saved video for {layout} to {video_path}")

        avg = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        rewards_by_layout[layout] = (avg, std, episode_rewards)

    print("\n--- Final Evaluation Summary ---")
    for layout, (avg, std, _) in rewards_by_layout.items():
        print(f"{layout}: {avg:.2f} +/- {std:.2f}")

    return rewards_by_layout


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(v) for v in obj]
    return obj