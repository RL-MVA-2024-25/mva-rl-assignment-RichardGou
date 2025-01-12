import os
import random

import torch
import numpy as np
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from env_hiv import HIVPatient  

# -------------------------------------------------------------------
# 1) Fonctions de création d'environnements
# -------------------------------------------------------------------
def make_env(domain_randomization=False):
    """
    Crée un environnement HIVPatient.
    Ajustez domain_randomization si vous voulez entraîner sur un patient aléatoire.
    """
    env = HIVPatient(domain_randomization=domain_randomization)
    env = TimeLimit(env, max_episode_steps=200)
    return env

def make_parallel_envs(num_envs=8, domain_randomization=False):
    """
    Crée un SubprocVecEnv (environnements parallèles) pour du multiprocessing.
    """
    return SubprocVecEnv([lambda: make_env(domain_randomization) for _ in range(num_envs)])


# -------------------------------------------------------------------
# 2) Hyperparamètres de la politique
# -------------------------------------------------------------------
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128],
                   vf=[256, 256])]
)

def linear_schedule(initial_lr: float, final_lr: float):

    def schedule(progress_remaining: float) -> float:

        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


# -------------------------------------------------------------------
# 3) Classe ProjectAgent
# -------------------------------------------------------------------
class ProjectAgent:
    def __init__(self, env=None, model_path="best_model.zip"):
        """
        Constructeur de l'agent.
        
        :param env: (optionnel) un vec_env (DummyVecEnv, SubprocVecEnv...) déjà créé.
                    Si None, on crée nous-mêmes un SubprocVecEnv + VecNormalize.
        :param model_path: chemin du modèle PPO à sauvegarder/charger.
        """
        self.model_path = model_path

        # Si aucun env n'est fourni, on le crée nous-mêmes
        if env is None:
            #  8 environnements parallèles
            vec_env = make_parallel_envs(num_envs=8, domain_randomization=False)
            # On applique la normalisation
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            vec_env = env

        # Instanciation du PPO
        self.model = PPO(
            policy=MlpPolicy,
            env=vec_env,
            verbose=1,
            tensorboard_log="./ppo_hiv_tensorboard/",
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(3e-3, 1e-3),
        )

    def load(self,training = False):
        """
        Charge le modèle (fichier .zip) et la configuration VecNormalize (fichier .pkl).
        """
        if os.path.exists(self.model_path):
            # On charge le modèle PPO
            self.model = PPO.load(self.model_path) 
            print(f"Modèle chargé depuis {self.model_path}")

            # Charger ensuite les stats de VecNormalize
            if os.path.exists("vec_normalize.pkl"):
                # Crée un nouvel env pour y injecter les stats
                dummy_env = make_parallel_envs(num_envs=8, domain_randomization=False)
                dummy_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
                # Si on veut continuer l'entraînement, on met training=True
                if training:
                    dummy_env.training = True
                    dummy_env.norm_reward = True
                else:
                    dummy_env.training = False
                    dummy_env.norm_reward = False

                # Associer cet env au modèle
                self.model.set_env(dummy_env)
                print("Statistiques VecNormalize chargées depuis vec_normalize.pkl")
            else:
                print("Aucune stats VecNormalize (vec_normalize.pkl) trouvée, utilisation d'un env par défaut.")
                # On recrée un env "neuf"
                dummy_env = make_parallel_envs(num_envs=16, domain_randomization=False)
                dummy_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
                self.model.set_env(dummy_env)
        else:
            print(f"Aucun modèle trouvé à {self.model_path}, on part de zéro.")

    def train(self, total_timesteps=500000):
        """
        Entraîne le modèle PPO pendant 'total_timesteps' itérations.
        """
        # Créer un env dédié à l'évaluation
        eval_env = DummyVecEnv([lambda: make_env(domain_randomization=False)])
        if os.path.exists("vec_normalize.pkl"):
            eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            print("Loaded VecNormalize stats pour évaluation.")
        else:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("Initialized fresh VecNormalize for evaluation.")

        # Callbacks : sauvegarde du meilleur modèle + checkpoints périodiques
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

        # Lance l'entraînement
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        # Sauvegarde du modèle
        self.model.save(self.model_path)
        print(f"Modèle final sauvegardé dans {self.model_path}")

        if isinstance(self.model.get_env(), VecNormalize):
            self.model.get_env().save("vec_normalize.pkl")
            print("Statistiques VecNormalize sauvegardées dans vec_normalize.pkl")
        else:
            print("Attention : l'env du modèle n'est pas un VecNormalize, pas de stats à sauvegarder.")

    def act(self, observation, use_random=False):
        """
        Sélection d'action à partir d'une observation donnée.
        :param observation: np.array (ou batch de vecteurs d'obs)
        :param use_random: bool, si True on choisit une action aléatoire
        """
        if use_random:
            return random.randint(0, 3)
        # Action déterministe depuis le modèle
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, path):
        """
        Sauvegarde manuellement le modèle + stats VecNormalize.
        """
        self.model.save(path)
        print(f"Modèle PPO sauvegardé dans : {path}")
        if isinstance(self.model.get_env(), VecNormalize):
            self.model.get_env().save("vec_normalize.pkl")
            print("Statistiques VecNormalize sauvegardées dans vec_normalize.pkl")
        else:
            print("Env du modèle non VecNormalize, pas de stats sauvegardées.")


def main():
    """
    Point d'entrée principal : crée l'agent, charge le modèle+stats si existants,
    puis lance l'entraînement.
    """
    
    agent = ProjectAgent(model_path="best_model.zip")
    agent.load(training = True)              
    agent.train(1000000)      



if __name__ == "__main__":
    main()
