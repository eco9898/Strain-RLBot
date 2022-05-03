from stable_baselines3 import PPO
import pathlib
from discrete_act import DiscreteAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "device": "auto",
            "n_envs": 1,
        }
        #print(str(_path) + '\\models\\exit_save')
        self.actor = PPO.load(str(_path) + '/models/exit_save', device='auto', custom_objects=custom_objects)
        self.parser = DiscreteAction()


    def act(self, obs, state):
        action = self.actor.predict(obs, state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)
        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")
