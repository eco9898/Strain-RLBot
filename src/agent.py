from stable_baselines3 import PPO
import pathlib
from discrete_act import DiscreteAction
import glob
import os.path
from trainer_classes import isKickoff
from rlgym.utils.gamestates import PlayerData, GameState

use_latest = True
use_kickoff = True
kickoff_override = False

class Agent:
    def __init__(self):
        global use_latest, use_kickoff, kickoff_override
        self.use_kickoff = use_kickoff
        self.kickoff_override = kickoff_override
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "device": "auto",
            "n_envs": 1,
        }
        if use_latest:
            if self.use_kickoff:
                try:
                    folder_path = str(_path) + '\\models\\kickoff'
                    file_type = r'\*.zip'
                    files = glob.glob(folder_path + file_type)
                    newest_kickoff_model = max(files, key=os.path.getctime)[0:-4]
                except:
                    self.use_kickoff = False
                    print("Failed to load newest kickoff")

            try:
                self.kickoffActor = PPO.load(newest_kickoff_model, device='auto', custom_objects=custom_objects)
                folder_path = str(_path) + '\\models\\match'
                file_type = r'\*.zip'
                files = glob.glob(folder_path + file_type)
                newest_match_model = max(files, key=os.path.getctime)[0:-4]
            except:
                self.kickoff_override = True
                print("Failed to load newest match")
            

            self.matchActor = PPO.load(newest_match_model, device='auto', custom_objects=custom_objects)
        else:
            if self.use_kickoff:
                try:
                    self.kickoffActor = PPO.load(str(_path) + '/models/kickoff/exit_save', device='auto', custom_objects=custom_objects)
                except:
                    self.use_kickoff = False
                    print("Failed to load exit kickoff")
            try:
                self.matchActor = PPO.load(str(_path) + '/models/match/exit_save', device='auto', custom_objects=custom_objects)
            except:
                self.kickoff_override = True
                print("Failed to load exit match")

        print ("Using latest:", use_latest)
        print ("Using kickoff:", self.use_kickoff)
        print ("Kickoff override:", self.kickoff_override)
        if self.use_kickoff:
            self.agent_used = "kickoff"
        else:
            self.agent_used = "match"
        print("Using agent: " + self.agent_used)
        self.parser = DiscreteAction()


    def act(self, player: PlayerData, obs, state: GameState):
        if (isKickoff(state) and self.use_kickoff) or self.kickoff_override:
            action = self.kickoffActor.predict(obs, state, deterministic=True)
            if self.agent_used != "kickoff":
                self.agent_used = "kickoff"
                print("Using agent: " + self.agent_used)
        else:
            action = self.matchActor.predict(obs, state, deterministic=True)
            if self.agent_used != "match":
                self.agent_used = "match"
                print("Using agent: " + self.agent_used)
        x = self.parser.parse_actions(action[0], state)
        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")

#switch out different models for kickoff
# Ive considered setting up different models with different roles, but thought it would be too difficult because of the kickoff, but if I had it set to load them in after kickoff if they dont have possession that might work
