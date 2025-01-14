import os
import random
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from env_hiv import HIVPatient  

#Environnement
def make_env(domain_randomization=True):
    def _init():
        env = HIVPatient(domain_randomization=domain_randomization)
        env = TimeLimit(env, max_episode_steps=200)
        return env
    return _init

#Hyperparamètres
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256],
                   vf=[512, 512])]
)
domain_randomization = True

def linear_schedule(initial_lr: float, final_lr: float):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

class ProjectAgent:
    def __init__(self):
        # Création de l'environnement vectorisé
        self.vec_env = SubprocVecEnv([make_env(domain_randomization) for _ in range(8)])  # 8 environnements parallèles
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # Normalisation

        # Instanciation du PPO
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
        self.model.learn(total_timesteps=total_timesteps)

        # Sauvegarde finale du modèle et de l'environnement
        self.save()
    
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
   
    agent.train(total_timesteps=2000000)

    agent.close()

if __name__ == "__main__":
    main()
