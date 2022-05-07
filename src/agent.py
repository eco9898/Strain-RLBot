from stable_baselines3 import PPO
import pathlib
from discrete_act import DiscreteAction
import glob
import os.path
from trainer_classes import isKickoff

use_latest = True
use_kickoff = False

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
            if use_kickoff:
                try:
                    folder_path = str(_path) + '\\models\\kickoff'
                    file_type = r'\*.zip'
                    files = glob.glob(folder_path + file_type)
                    newest_kickoff_model = max(files, key=os.path.getctime)[0:-4]
                except:
                    use_kickoff = False

            self.kickoffActor = PPO.load(newest_kickoff_model, device='auto', custom_objects=custom_objects)
            folder_path = str(_path) + '\\models\\match'
            file_type = r'\*.zip'
            files = glob.glob(folder_path + file_type)
            newest_match_model = max(files, key=os.path.getctime)[0:-4]

            self.matchActor = PPO.load(newest_match_model, device='auto', custom_objects=custom_objects)
        else:
            self.kickoffActor = PPO.load(str(_path) + '/models/kickoff/exit_save', device='auto', custom_objects=custom_objects)
            self.matchActor = PPO.load(str(_path) + '/models/match/exit_save', device='auto', custom_objects=custom_objects)
        self.parser = DiscreteAction()


    def act(self, obs, state):
        if isKickoff() and use_kickoff:
            action = self.kickoffActor.predict(obs, state, deterministic=True)
        else:
            action = self.matchActor.predict(obs, state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)
        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")

#switch out different models for kickoff
# Ive considered setting up different models with different roles, but thought it would be too difficult because of the kickoff, but if I had it set to load them in after kickoff if they dont have possession that might work
