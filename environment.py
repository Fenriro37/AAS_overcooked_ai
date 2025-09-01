import random
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

class GeneralizedOvercooked:
    def __init__(self, layouts, info_level=0, horizon=400):
        self.envs = []
        for layout in layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append((layout, env))

        self.cur_layout, self.cur_env = self.envs[0]
        self.observation_space, self.action_space = self.cur_env.observation_space, self.cur_env.action_space
        self.horizon = horizon
        self.rewards_per_layout = {layout: [] for layout in layouts}

    def reset(self):
        idx = random.randint(0, len(self.envs) - 1)
        self.cur_layout, self.cur_env = self.envs[idx]
        return self.cur_env.reset()

    def step(self, *args):
        return self.cur_env.step(*args)

    def render(self, *args):
        return self.cur_env.render(*args)

    def get_layout_name(self):
        return self.cur_layout

    def sample_layout(self, strategy="uniform", epsilon=0.1):
        if strategy == "uniform":
            idx = random.randint(0, len(self.envs) - 1)
        elif strategy == "weighted":
            avg_rewards = [np.mean(self.rewards_per_layout[layout][-50:]) if self.rewards_per_layout[layout] else 0.0 for layout, _ in self.envs]
            avg_rewards = np.array(avg_rewards, dtype=np.float64)
            weights = 1.0 / (avg_rewards + 1.0)
            if weights.sum() == 0 or np.any(np.isnan(weights)):
                weights = np.ones(len(self.envs))
            weights = (1 - epsilon) * (weights / weights.sum()) + (epsilon / len(self.envs))
            idx = np.random.choice(len(self.envs), p=weights)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        self.cur_layout, self.cur_env = self.envs[idx]
        return idx

    def custom_reset(self, strategy=None, idx=None):
        if idx is not None:
            self.cur_layout, self.cur_env = self.envs[idx]
        elif strategy is not None:
            self.sample_layout(strategy=strategy)
        else:
            idx = random.randint(0, len(self.envs) - 1)
            self.cur_layout, self.cur_env = self.envs[idx]
        return self.cur_env.reset()