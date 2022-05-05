from stable_baselines3 import PPO
import pathlib
from discrete_act import DiscreteAction
import glob
import os.path

use_latest = True

class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "device": "auto",
            "n_envs": 1,
        }
        if use_latest:
            folder_path = str(_path) + '\\models'
            file_type = r'\*zip'
            files = glob.glob(folder_path + file_type)
            newest_model = max(files, key=os.path.getctime)[0:-4]

            self.actor = PPO.load(newest_model, device='auto', custom_objects=custom_objects)
        else:
            self.actor = PPO.load(str(_path) + '/models/exit_save', device='auto', custom_objects=custom_objects)
        self.parser = DiscreteAction()


    def act(self, obs, state):
        action = self.actor.predict(obs, state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)
        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")

#switch out different models for kickoff
# Ive considered setting up different models with different roles, but thought it would be too difficult because of the kickoff, but if I had it set to load them in after kickoff if they dont have possession that might work
