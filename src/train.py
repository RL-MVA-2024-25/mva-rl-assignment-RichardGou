import os
import random
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.monitor import Monitor
from env_hiv import HIVPatient  

def make_env(domain_randomization=True, log_dir=None):
    def _init():
        env = HIVPatient(domain_randomization=domain_randomization)
        env = TimeLimit(env, max_episode_steps=200)
        if log_dir is not None:
            env = Monitor(env, filename=os.path.join(log_dir, f"env_{random.randint(0, 10000)}.log"))
        return env
    return _init

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256],
                   vf=[512, 512])]
)
domain_randomization = True

def linear_schedule(initial_lr: float, final_lr: float):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.save_path_env = os.path.join(log_dir, "vec_normalize.pkl")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
            except FileNotFoundError:
                return True
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    vec_env = self.model.get_vec_normalize_env()
                    if vec_env is not None:
                        vec_env.save(self.save_path_env)
                        if self.verbose > 0:
                            print(f"Environnement normalisé sauvegardé dans : {self.save_path_env}")
        return True

class ProjectAgent:
    def __init__(self, log_dir="./logs/"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.vec_env = SubprocVecEnv([make_env(domain_randomization, log_dir=self.log_dir) for _ in range(8)])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        self.model = PPO(
            policy=MlpPolicy,
            env=self.vec_env,
            verbose=1,
            tensorboard_log="./ppo_hiv_tensorboard/",
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(3e-3, 1e-3),
        )
        
    def load(self, model_path="best_model.zip", env_path="vec_normalize.pkl"):
        if os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.vec_env)
            print(f"Modèle chargé depuis {model_path}")
        else:
            print(f"Fichier modèle {model_path} non trouvé.")
            return

        if os.path.exists(env_path):
            self.vec_env = VecNormalize.load(env_path, self.vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            print(f"Environnement normalisé chargé depuis {env_path}")
        else:
            print(f"Fichier environnement {env_path} non trouvé.")
    
    def train(self, total_timesteps=500000):
        callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir, verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def act(self, observation, use_random=False):
        if use_random:
            return random.randint(0, 3)
        
        observation = self.vec_env.normalize_obs(observation)
        action, _states = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path_model="best_model.zip", path_env="vec_normalize.pkl"):
        self.model.save(path_model)
        print(f"Modèle PPO sauvegardé dans : {path_model}")
        self.vec_env.save(path_env)
        print(f"Environnement normalisé sauvegardé dans : {path_env}")
    
    def close(self):
        self.vec_env.close()
        print("Environnement fermé.")

def main():
    agent = ProjectAgent()
    agent.train(total_timesteps=5000000)
    agent.close()

if __name__ == "__main__":
    main()
