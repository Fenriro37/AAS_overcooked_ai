import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models import create_network

class MAPPOBuffer:
    def __init__(self, obs_shape, state_shape, buffer_size, n_agents):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.int32)
        
        self.observations = np.zeros((buffer_size, n_agents, *obs_shape), dtype=np.int32)
        self.actions = np.zeros((buffer_size, n_agents), dtype=np.int32)
        self.log_probs = np.zeros((buffer_size, n_agents), dtype=np.float32)
        
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=np.int32)
        
        self.values = np.zeros((buffer_size, n_agents), dtype=np.float32)
        
        # Computed later
        self.advantages = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_agents), dtype=np.float32)
        
        self.pos = 0

    def store(self, obs, state, action, reward, done, value, log_prob):

        self.observations[self.pos] = obs
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos = (self.pos + 1) % self.buffer_size

    def get(self):

        obs = self.observations.reshape(self.buffer_size * self.n_agents, -1)
        states = self.states.reshape(self.buffer_size, -1)
        actions = self.actions.reshape(self.buffer_size * self.n_agents)
        log_probs = self.log_probs.reshape(self.buffer_size * self.n_agents)
        values = self.values.reshape(self.buffer_size * self.n_agents)
        advantages = self.advantages.reshape(self.buffer_size * self.n_agents)
        returns = self.returns.reshape(self.buffer_size * self.n_agents)
        rewards = self.rewards.reshape(self.buffer_size * self.n_agents)
        dones = self.dones.reshape(self.buffer_size)

        return {
            "observations": obs,
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
            "dones": dones
        }
    
    def clear(self):
        self.states.fill(0)
        self.observations.fill(0)
        self.actions.fill(0)
        self.log_probs.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.pos = 0

    def compute_advantages(self, gamma, lam):
        values = self.values         
        rewards = self.rewards       
        dones = self.dones           
        
        for agent_idx in range(self.n_agents):
            agent_rewards = rewards[:, agent_idx]     
            agent_values = values[:, agent_idx]  
            gae = 0

            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 0.0
                    next_value_step = 0.0
                else:
                    next_non_terminal = 1.0 - dones[step]
                    next_value_step = agent_values[step + 1]

                delta = agent_rewards[step] + gamma * next_value_step * next_non_terminal - agent_values[step]
                gae = delta + gamma * lam * next_non_terminal * gae

                self.advantages[step, agent_idx] = gae
                self.returns[step, agent_idx] = gae + agent_values[step]

class MAPPO:
    def __init__(self, env,max_timesteps, strategy='weighted', num_agents=2,
                 buffer_size=4000, 
                 lr_policy=3e-4, lr_value=1e-3, 
                 epochs=7, 
                 batch_size=256, 
                 clip_ratio=0.2, 
                 entropy_coef=0.01, 
                 value_coef=0.01,
                 gamma = 0.99, 
                 lam=0.95,
                 shaping_coef=1.0,shaping_decay=None,decay_until="half_run",
                 dense_units=128,net_depth=2,activation="tanh",
                 synched=True):
        self.env = env
        self.strategy = strategy
        self.num_agents = num_agents
        self.observation_space = env.observation_space.shape
        self.action_space = int(env.action_space.n)

        self.policy_net = create_network(
            input_shape=self.observation_space,
            model_type='policy',
            num_actions=self.action_space,
            dense_units=dense_units,
            depth=net_depth,
            activation=activation
        )

        state_dim = self.observation_space[0] * self.num_agents
        state_shape_tuple = (state_dim,)
        self.value_net = create_network(
            input_shape=state_shape_tuple,
            model_type='value',
            dense_units=dense_units,
            depth=net_depth,
            activation=activation
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(lr_policy)
        self.value_optimizer = tf.keras.optimizers.Adam(lr_value)

        #buffer initialization
        self.episode_length = env.horizon
        self.rollout_length = buffer_size
        self.episodes_in_rollout = buffer_size // self.episode_length
        
        self.max_timesteps = max_timesteps
        self.total_rollouts = self.max_timesteps // self.rollout_length

        self.buffer = MAPPOBuffer(
            obs_shape=self.observation_space,
            state_shape=state_shape_tuple,
            buffer_size=buffer_size,
            n_agents=num_agents
        )
        #hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.epochs=epochs
        self.batch_size=batch_size
        self.clip_ratio=clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        #shaping rewards parameters
        self.shaping_coef = shaping_coef
        if shaping_decay is not None:
            self.shaping_decay = shaping_decay
        else: 
            if decay_until is None or decay_until == "half_run":
                self.decay_until = self.total_rollouts // 2
            elif decay_until == "full_run":
                self.decay_until = self.total_rollouts
            else:
                self.decay_until = decay_until

            #Precompute shaping_decay so that shaping_coef reaches 0 at decay_until
            self.shaping_decay = shaping_coef / self.decay_until
        self.flag = True

        self.synched = synched

    def collect_rollout(self):
        obs_dict = self.env.custom_reset(idx=0)  
        obs = obs_dict["both_agent_obs"]  
        state = np.concatenate(obs, axis=-1)

        # --- Episode tracking ---
        episode_rewards = []         
        episode_rewards_per_agent = [] 
        episode_lengths = []   
        episode_layouts = []           
        

        current_ep_rewards = np.zeros(self.num_agents, dtype=np.float32)
        current_ep_length = 0

        for _ in range(self.rollout_length):
            obs_batch = np.stack(obs) 

            probs = self.policy_net(obs_batch)  
            actions = tf.random.categorical(tf.math.log(probs), 1).numpy().flatten()
            log_probs = tf.math.log(tf.gather(probs, actions, batch_dims=1)).numpy()

            value = self.value_net(state[None, :]).numpy().flatten() #fancy way of adding a batch dimension

            next_obs_dict, rewards, done, info_dict = self.env.step(actions) 
            shaped_reward = np.sum(info_dict.get("shaped_r_by_agent", 0))
            rewards = rewards + self.shaping_coef * shaped_reward
            
            next_obs = next_obs_dict["both_agent_obs"]
            next_state = np.concatenate(next_obs, axis=-1)

            self.buffer.store(
                obs=np.array(obs),
                state=state,
                action=actions,
                reward=np.array(rewards),
                done=float(done),
                value=value,
                log_prob=log_probs
            )

            current_ep_rewards += np.array(rewards)
            current_ep_length += 1

            obs, state = next_obs, next_state

            if done:
                ep_total = float(current_ep_rewards.sum())
                episode_rewards.append(ep_total)
                episode_rewards_per_agent.append(tuple(current_ep_rewards))
                episode_lengths.append(current_ep_length)
                episode_layouts.append(self.env.get_layout_name())

                # Update layout reward history used by "weighted" sampling
                self.env.rewards_per_layout[self.env.get_layout_name()].append(ep_total)

                current_ep_rewards[:] = 0
                current_ep_length = 0

                obs_dict = self.env.custom_reset(strategy=self.strategy)
                obs = obs_dict["both_agent_obs"]
                state = np.concatenate(obs, axis=-1)
                
        self.buffer.compute_advantages(self.gamma, self.lam)
        
        self.shaping_coef = max(0.0, self.shaping_coef - self.shaping_decay)
        
        if self.shaping_coef == 0.0 and self.flag:
            print(f"Shaping coefficient reached 0. Stopping shaping rewards.")
            self.flag = False

        rollout_stats = {
            "episode_rewards": episode_rewards,
            "episode_rewards_per_agent": episode_rewards_per_agent,
            "episode_lengths": episode_lengths,
            "episode_layouts": episode_layouts    

        }
        return rollout_stats

    def get_synced_batches(self, states, observations, actions, log_probs, advantages, returns):
            num_steps = states.shape[0]
            indices = np.random.permutation(num_steps)  

            for start in range(0, num_steps, self.batch_size):
                end = min(start + self.batch_size, num_steps)
                batch_t = indices[start:end]  

                # Critic batch (state-level)
                batch_states = tf.constant(states[batch_t], dtype=tf.float32)

                # Actor batch (expand timesteps to agents)
                agent_indices = np.array([
                    t * self.num_agents + agent for t in batch_t for agent in range(self.num_agents)
                ])

                batch_obs       = tf.constant(observations[agent_indices], dtype=tf.float32)
                batch_actions   = tf.constant(actions[agent_indices], dtype=tf.int32)
                batch_log_probs = tf.constant(log_probs[agent_indices], dtype=tf.float32)
                batch_adv       = tf.constant(advantages[agent_indices], dtype=tf.float32)

                state_returns = returns[::self.num_agents]  
                batch_returns = tf.constant(state_returns[batch_t], dtype=tf.float32)

                yield (batch_states,
                    batch_obs, batch_actions, batch_log_probs,
                    batch_adv, batch_returns)
                
    def get_independent_batches(self, states, observations, actions, log_probs, advantages, returns):

        num_steps = states.shape[0]          
        num_agent_samples = observations.shape[0]  

        state_indices = np.random.permutation(num_steps)              
        agent_indices = np.random.permutation(num_agent_samples)     

        for start in range(0, num_steps, self.batch_size):
            end = min(start + self.batch_size, num_steps)
            batch_t = state_indices[start:end]

            batch_states = tf.constant(states[batch_t], dtype=tf.float32)

            state_returns = returns[::self.num_agents]  
            batch_returns = tf.constant(state_returns[batch_t], dtype=tf.float32)

            yield ("critic", batch_states, batch_returns)

        for start in range(0, num_agent_samples, self.batch_size):
            end = min(start + self.batch_size, num_agent_samples)
            batch_agent = agent_indices[start:end]

            batch_obs       = tf.constant(observations[batch_agent], dtype=tf.float32)
            batch_actions   = tf.constant(actions[batch_agent], dtype=tf.int32)
            batch_log_probs = tf.constant(log_probs[batch_agent], dtype=tf.float32)
            batch_adv       = tf.constant(advantages[batch_agent], dtype=tf.float32)

            yield ("actor", batch_obs, batch_actions, batch_log_probs, batch_adv)

    def update(self):
        data = self.buffer.get()
        observations = data["observations"]
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(self.epochs):
            gen = self.get_synced_batches(states, observations, actions, old_log_probs,
                                          advantages, returns)

            for mb_states, mb_obs, mb_actions, mb_old_log_probs, mb_adv, mb_returns in gen:
                # --- Policy update ---
                with tf.GradientTape() as tape_p:
                    action_probs = self.policy_net(mb_obs)
                    action_mask = tf.one_hot(mb_actions, depth=action_probs.shape[-1], dtype=tf.float32)
                    selected_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                    new_log_probs = tf.math.log(selected_probs + 1e-8)

                    ratio = tf.exp(new_log_probs - mb_old_log_probs)
                    unclipped = ratio * mb_adv
                    clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                    entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=-1))
                    loss_pi = policy_loss - self.entropy_coef * entropy

                grads_pi = tape_p.gradient(loss_pi, self.policy_net.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grads_pi, self.policy_net.trainable_variables))

                # --- Value update ---
                with tf.GradientTape() as tape_v:
                    values = tf.squeeze(self.value_net(mb_states), axis=-1)
                    value_loss = tf.reduce_mean(tf.square(mb_returns - values))
                    loss_v = self.value_coef * value_loss

                grads_v = tape_v.gradient(loss_v, self.value_net.trainable_variables)
                self.value_optimizer.apply_gradients(zip(grads_v, self.value_net.trainable_variables))

        return {
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(entropy.numpy())
        }
    
    def update_independet(self):
        data = self.buffer.get()
        observations = data["observations"]
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(self.epochs):
            gen = self.get_independent_batches(states, observations, actions, old_log_probs,
                                            advantages, returns)
            
            for batch in gen:
                if batch[0] == "actor":
                    _, mb_obs, mb_actions, mb_old_log_probs, mb_adv = batch

                    with tf.GradientTape() as tape_p:
                        action_probs = self.policy_net(mb_obs)
                        action_mask = tf.one_hot(mb_actions, depth=action_probs.shape[-1], dtype=tf.float32)
                        selected_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
                        new_log_probs = tf.math.log(selected_probs + 1e-8)

                        ratio = tf.exp(new_log_probs - mb_old_log_probs)
                        unclipped = ratio * mb_adv
                        clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                        policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=-1))
                        loss_pi = policy_loss - self.entropy_coef * entropy

                    grads_pi = tape_p.gradient(loss_pi, self.policy_net.trainable_variables)
                    self.policy_optimizer.apply_gradients(zip(grads_pi, self.policy_net.trainable_variables))

                elif batch[0] == "critic":
                    _, mb_states, mb_returns = batch

                    with tf.GradientTape() as tape_v:
                        values = tf.squeeze(self.value_net(mb_states), axis=-1)
                        value_loss = tf.reduce_mean(tf.square(mb_returns - values))
                        loss_v = self.value_coef * value_loss

                    grads_v = tape_v.gradient(loss_v, self.value_net.trainable_variables)
                    self.value_optimizer.apply_gradients(zip(grads_v, self.value_net.trainable_variables))

        return {
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(entropy.numpy())
        }

    def train(self, save_dir="models"):
        timesteps = 0
        all_episode_rewards = []
        all_episode_environments = []
        all_policy_losses = []
        all_value_losses = []
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with tqdm(total=self.max_timesteps, desc="Training Progress") as pbar:
            while timesteps < self.max_timesteps:
                rollout_stats = self.collect_rollout()
                if self.synched:
                    update_stats = self.update()
                else:
                    update_stats = self.update_independet()

                self.buffer.clear()

                if rollout_stats["episode_rewards"]:
                    all_episode_rewards.extend(rollout_stats["episode_rewards"])
                    all_episode_environments.extend(rollout_stats["episode_layouts"])
                all_policy_losses.append(update_stats["policy_loss"])
                all_value_losses.append(update_stats["value_loss"])

                pbar.update(self.rollout_length)
                timesteps += self.rollout_length
                if all_episode_rewards:
                    postfix = {
                        "Step": timesteps,
                        f"AvgR({self.episodes_in_rollout})": f"{np.mean(all_episode_rewards[-self.episodes_in_rollout:]):.2f}",
                        "pi_loss": f"{update_stats['policy_loss']:.2f}",
                        "v_loss": f"{update_stats['value_loss']:.2f}",
                    }
                    pbar.set_postfix(postfix)
        
        self.policy_net.save(os.path.join(save_dir, "policy_net.keras"))
        self.value_net.save(os.path.join(save_dir, "value_net.keras"))
        print(f"Models saved to {save_dir}")

        training_stats = {
            "episode_rewards": all_episode_rewards,
            "episode_layouts": all_episode_environments,
            "policy_losses": all_policy_losses,
            "value_losses": all_value_losses,
        }
        return training_stats