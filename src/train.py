from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Function to create the environment
def make_env():
    env = HIVPatient(domain_randomization=False)
    env = TimeLimit(env, max_episode_steps=200)
    return env

# Initialize and wrap the environment
vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

class ProjectAgent:
    def __init__(self, env=vec_env, model_path="best_model.zip"):
        self.model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log="./ppo_hiv_tensorboard/"
        )
        self.model_path = model_path

    def train(self, total_timesteps=1000000):
        # Define evaluation environment
        eval_env = DummyVecEnv([make_env])
        
        # Load VecNormalize stats if available
        if os.path.exists("./vec_normalize.pkl"):
            eval_env = VecNormalize.load("./vec_normalize.pkl", eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            print("Loaded VecNormalize stats for evaluation.")
        else:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
            print("Initialized VecNormalize for evaluation.")

        # Create EvalCallback to save the best model
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=10000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )

        # Create CheckpointCallback to save checkpoints periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path='./checkpoints/',
            name_prefix='ppo_hiv_checkpoint'
        )

        # Combine callbacks
        callback = CallbackList([eval_callback, checkpoint_callback])

        # Train the model with callbacks
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save the final model
        self.model.save(self.model_path)
        print(f"Final model saved to {self.model_path}")

        # Save the VecNormalize statistics
        vec_env.save("vec_normalize.pkl")
        print("VecNormalize statistics saved to vec_normalize.pkl")

    def act(self, observation, use_random=False):
        if use_random:
            return random.randint(0, 3)
        # Get action from the trained model
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save(self, path):
        self.model.save(path)
        vec_env.save("vec_normalize.pkl")
        print(f"Model and VecNormalize stats saved to {path} and vec_normalize.pkl respectively.")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, env=vec_env)
            if os.path.exists("vec_normalize.pkl"):
                vec_env.load("vec_normalize.pkl")
                print(f"Model loaded from {self.model_path} and VecNormalize stats loaded from vec_normalize.pkl")
            else:
                print("Model loaded, but VecNormalize stats file vec_normalize.pkl not found.")
        else:
            print(f"No model found at {self.model_path}, starting fresh.")

if __name__ == "__main__":
    # Initialize agent
    agent = ProjectAgent()

    # Load existing model and normalization stats if available
    agent.load()

    # Train the agent
    agent.train(total_timesteps=1000000)
