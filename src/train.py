import os
import random
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

def make_env():
    env = HIVPatient(domain_randomization=False)
    env = TimeLimit(env, max_episode_steps=200)
    return env

def make_parallel_envs(num_envs=8):
    return SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])

policy_kwargs = dict(net_arch=[256, 256])

class ProjectAgent:
    def __init__(self, env, model_path="best_model.zip"):
        self.model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log="./ppo_hiv_tensorboard/",
            policy_kwargs=policy_kwargs,
        )
        self.model_path = model_path

    def train(self, total_timesteps=1000000):
        # Create a separate env for evaluation
        eval_env = DummyVecEnv([make_env])

        if os.path.exists("./vec_normalize.pkl"):
            eval_env = VecNormalize.load("./vec_normalize.pkl", eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            print("Loaded VecNormalize stats for evaluation.")
        else:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("Initialized VecNormalize for evaluation.")

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=10000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path='./checkpoints/',
            name_prefix='ppo_hiv_checkpoint'
        )

        callback = CallbackList([eval_callback, checkpoint_callback])

        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        self.model.save(self.model_path)
        print(f"Final model saved to {self.model_path}")

        # Save the VecNormalize env
        self.model.get_env().save("vec_normalize.pkl")
        print("VecNormalize statistics saved to vec_normalize.pkl")

    def act(self, observation, use_random=False):
        if use_random:
            return random.randint(0, 3)
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, path):
        self.model.save(path)
        self.model.get_env().save("vec_normalize.pkl")
        print(f"Model and VecNormalize stats saved to {path} and vec_normalize.pkl respectively.")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            # if you want, re-set the environment or re-wrap with VecNormalize
            # but that means you need to create it again:
            print(f"Model loaded from {self.model_path}")
            if os.path.exists("vec_normalize.pkl"):
                self.model.set_env(VecNormalize.load("vec_normalize.pkl", SubprocVecEnv([lambda: make_env()])))
                # or dummy env if not training
                print("VecNormalize stats loaded from vec_normalize.pkl")
            else:
                print("vec_normalize.pkl not found.")
        else:
            print(f"No model found at {self.model_path}, starting fresh.")

def main():
    # Everything that spawns processes must be inside main:
    vec_env = make_parallel_envs(num_envs=8)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize agent
    agent = ProjectAgent(env=vec_env)
    agent.load()

    # Train
    agent.train(total_timesteps=1000000)

if __name__ == "__main__":
    main()
