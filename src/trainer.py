from time import sleep, time
import multiprocessing
import os, signal
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.terminal_conditions.common_conditions import *
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import CombinedReward

from advanced_padder import AdvancedObsPadder
from discrete_act import DiscreteAction
from trainer_classes import *

MAX_INSTANCES_NO_PAGING = 5
WAIT_TIME_NO_PAGING = 22
WAIT_TIME_PAGING = 40

num_instances = 5
kickoff_instances = num_instances // 3
match_instances = num_instances - kickoff_instances
models = [["kickoff", kickoff_instances], ["match", match_instances]]

paging = False
if num_instances > MAX_INSTANCES_NO_PAGING:
    paging = True
wait_time=WAIT_TIME_NO_PAGING
if paging:
    wait_time=WAIT_TIME_PAGING

def killRL(targets: List = []):
    PIDs = getRLInstances()
    while len(PIDs) > 0:
        pid = PIDs.pop()
        if pid in targets:
            print(">>Killing RL instance", pid["pid"])
            try:
                os.kill(pid["pid"], signal.SIGTERM)
            except:
                print(">>Failed")

def start_training(send_messages: multiprocessing.Queue, model_args: List):
    global paging, wait_time
    name = model_args[0]
    num_instances = model_args[1]
    frame_skip = 12          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120/frame_skip #120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    team_size = 3
    self_play = True
    if self_play:
        agents_per_match = 2*team_size
    else:
        agents_per_match = team_size
    print(">>>Wait time:        ", wait_time)
    print(">>># of instances:   ", num_instances)
    print(">>>Paging:           ", paging)
    n_env = agents_per_match * num_instances
    print(">>># of env:         ", n_env)
    target_steps = 1_000_000
    steps = target_steps//n_env #making sure the experience counts line up properly
    print(">>>Steps:            ", steps)
    batch_size = (100_000//(steps))*(steps) #getting the batch size down to something more manageable - 80k in this case at 5 instances, but 25k at 16 instances
    print(">>>Batch size:       ", batch_size)
    training_interval = 25_000_000
    print(">>>Training interval:", training_interval)
    mmr_save_frequency = 50_000_000
    print(">>>MMR frequency:    ", mmr_save_frequency)
    send_messages.put(1)
    attackRewards = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            RewardIfClosestToBall(AlignBallGoal(0,1), True),
        ),
        (2.0, 0.2, 1.0, 1.0, 0.8))

    defendRewards = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0)
        ),
        (1.0, 0.2, 1.0, 1.0, 1.5))

    lastManRewards = CombinedReward(
        (
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            AlignBallGoal(1,0),
            LiuDistancePlayerToGoalReward(),
            ConstantReward()
        ),
        (2.0, 1.0, 0.6, 0.2))

    kickoffRewards = CombinedReward(
        (
            RewardIfClosestToBall(
                CombinedReward(
                    (
                        VelocityPlayerToBallReward(),
                        AlignBallGoal(0,1),
                        LiuDistancePlayerToBallReward(),
                        FlipReward(),
                        useBoost()
                    ),
                    (20.0, 1.0, 2.0, 2.0, 5.0)
                ),
                team_only=True
            ),
            RewardIfMidFromBall(defendRewards),
            RewardIfFurthestFromBall(lastManRewards),
            TeamSpacingReward(),
            pickupBoost()
        ),
        (2.0, 1.0, 1.5, 1.0, 0.4))

    def exit_save(model, name: str):
        model.save("src/models/" + name + "/exit_save")

    def load_save(name: str, env, steps, batch_size, MlpPolicy, gamma):
        try:
            model = PPO.load(
                "src/models/" + name + "/exit_save",
                env,
                device="auto",
                #custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
                #If you need to adjust parameters mid training, you can use the below example as a guide
                custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "_last_obs": None}
            )
            print(">>>Loaded previous exit save.")
        except:
            print(">>>No saved model found, creating new model.")
            from torch.nn import Tanh
            policy_kwargs = dict(
                activation_fn=Tanh,
                net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
            )

            return PPO(
                MlpPolicy,
                env,
                n_epochs=10,                 # PPO calls for multiple epochs
                policy_kwargs=policy_kwargs,
                learning_rate=5e-5,          # Around this is fairly common for PPO
                ent_coef=0.01,               # From PPO Atari
                vf_coef=1.,                  # From PPO Atari
                gamma=gamma,                 # Gamma as calculated using half-life
                verbose=3,                   # Print out all the info as we're going
                batch_size=batch_size,             # Batch size as high as possible within reason
                n_steps=steps,                # Number of steps to perform before optimizing network
                tensorboard_log="src/logs/" + name,  # `tensorboard --logdir out/logs` in terminal to see graphs
                device="auto"                # Uses GPU if available
            )

    def get_base_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size, #amount of bots per team
            tick_skip=frame_skip,
            self_play=self_play, #play against its self
            obs_builder=AdvancedObsPadder(3),  # Not that advanced, good default
            action_parser=DiscreteAction(),  # Discrete > Continuous don't @ me match._reward_fn = CombinedReward(
            reward_function= CombinedReward((), ()),
            terminal_conditions = [TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
            state_setter = RandomState()  # Resets to random
        )

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        match: Match = get_base_match()
        match._reward_fn = CombinedReward(
            (
                RewardIfAttacking(attackRewards),
                RewardIfDefending(defendRewards),
                RewardIfLastMan(lastManRewards),
                VelocityReward(),
                FaceBallReward(),
                EventReward(
                    team_goal=100.0,
                    goal=20.0,
                    concede=-120.0,
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                JumpTouchReward(),
                TouchBallReward(1.2),
                TeamSpacingReward(1500),
                FlipReward(),
                SaveBoostReward(),
                pickupBoost(),
                useBoost()
            ),
            (1.0, 1.0, 3.0, 10.0, 1.0, 1.0, 1.5, 1.0, 5.0, 0.6, 1.0, 1.6, np.sqrt(30)))
        match._terminal_conditions = [TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()]
        match._state_setter = RandomState()  # Resets to random
        return match

    def get_kickoff():  # Need to use a function so that each instance can call it and produce their own objects
        match: Match = get_base_match()
        match._reward_fn = CombinedReward(
            (
                kickoffRewards,
                VelocityReward(),
                FaceBallReward(),
                EventReward(
                    team_goal=100.0,
                    goal=20.0,
                    concede=-120.0,
                    shot=10.0,
                    save=30.0,
                    demo=12.0,
                ),
                JumpTouchReward(),
                TouchBallReward(1.2),
                TeamSpacingReward(1500),
                FlipReward(),
                SaveBoostReward(),
                pickupBoost(),
                useBoost(),
            ),
            (5.0, 10.0, 3.0, 1.0, 1.5, 1.0, 5.0, 0.6, 1.0, 1.6, np.sqrt(30)))
        #time out after 50 seconds encourage kickoff
        match._terminal_conditions = [TimeoutCondition(fps * 50), NoTouchTimeoutCondition(fps * 20), GoalScoredCondition()]
        match._state_setter = DefaultState()  # Resets to kickoff position
        return match
    

    while True:
        try:
            if name == "kickoff":
                match = get_kickoff
            else:
                match = get_match
            env = SB3MultipleInstanceEnv(match, model_args[1], force_paging=paging, wait_time=wait_time) #or 40            # Start instances, waiting 60 seconds between each
            env = VecCheckNan(env)                                # Optional
            env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
            env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

            model = load_save(model_args[0], env, steps, batch_size, MlpPolicy, gamma)

            # Save model every so often
            # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
            # This saves to specified folder with a specified name
            callback = CheckpointCallback(round(1_000_000 / env.num_envs), save_path="src/models/" + name, name_prefix="rl_model") # backup every 5 mins
            #1mill per 8/9 minutes at 5 instances?

            try:
                mmr_model_target_count = model.num_timesteps + (mmr_save_frequency - (model.num_timesteps % mmr_save_frequency)) #current steps + remaing steps until mmr model
                while True:
                    new_training_interval = training_interval - (model.num_timesteps % training_interval) # remaining steps to train interval
                    print(">>>Training for %s timesteps" % new_training_interval)
                    send_messages.put(2)
                    #may need to reset timesteps when you're running a different number of instances than when you saved the model
                    #subprocess learning
                    model.learn(new_training_interval, callback=callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
                    exit_save(model)
                    if model.num_timesteps >= mmr_model_target_count:
                        model.save(f"src/mmr_models/{model_args[0]}/{model.num_timesteps}")
                        mmr_model_target_count += mmr_save_frequency

            except KeyboardInterrupt:
                print(">>>Exiting training")
            print(">>>Saving model")
            exit_save(model)
            print(">>>Save complete")
            break
        except KeyboardInterrupt:
            break
        #except:
        #    pass
def trainingStarter(send_messages: multiprocessing.Queue, model_args):
    instances = model_args[1]
    done = False
    RLinstances = []
    while not done:
        initial_RLProc = len(getRLInstances())
        print(">>Initial instances:", initial_RLProc)
        print(">>Starting trainer")
        receive_messages = multiprocessing.Queue()
        trainer = multiprocessing.Process(target=start_training, args=[receive_messages, model_args])
        trainer.start()
        count = 0
        #Wait until setup is printed
        while receive_messages.empty() and trainer.is_alive():
            sleep(1)
        receive_messages.get()
        start = time()
        while count < instances  and trainer.is_alive():
            print(">>Parsing instance:" , (count + 1))
            curr_count = 0
            while (time() - start) // wait_time <= count:
                curr_count = len(getRLInstances()) - initial_RLProc
                if curr_count != count:
                    break
                sleep(1)
            if curr_count > count:
                count = curr_count
                print(">>Instances found:" , count)
                RLinstances.append(getRLInstances()[-1])
                if count == instances:
                    break
            else:
                break
        done = False
        if count == instances:
            print(">>Waiting to start")
            start = time()
            while (time() - start) < wait_time * 2 and trainer.is_alive():
                if not receive_messages.empty():
                    break
            if not receive_messages.empty() and trainer.is_alive():
                if receive_messages.get() == 2:
                    done = True
        if count != instances or not done:
            print(">>Killing trainer")
            trainer.terminate()
            killRL(RLinstances)
        else:
            minimiseRL()
            try:
                print(">>Finished parsing trainer")
                send_messages.put(1)
                while trainer.is_alive():
                    sleep(1)
                #trainer died restart loop
                killRL(RLinstances)
            except KeyboardInterrupt:
                #trainer will shut down and save, please wait
                while trainer.is_alive():
                    sleep(0.1)
                break
    killRL(RLinstances)

def start_starter(messages: List[multiprocessing.Queue], starters: List[multiprocessing.Process], model_args):
    done = False
    print(">Starting trainer: " + model_args[0])
    while not done:
        messages.append(multiprocessing.Queue())
        starters.append(multiprocessing.Process(target=trainingStarter, args=[messages[-1], model_args]))
        starters[-1].start()
        #wait to open RL instances
        start = time()
        while (time() - start) < wait_time * (model_args[1] + 2) and starters[-1].is_alive():
            if not messages[-1].empty():
                break
        if not messages[-1].empty():
            if messages[-1].get() == 1:
                done = True
        if not done:
            print(">Restarting trainer: " + model_args[0])
    print(">Training started: " + model_args[0])

if __name__ == "__main__":
    messages: List[multiprocessing.Queue] = []
    starters: List[multiprocessing.Process] = []
    for model_args in models:
        start_starter(messages, starters, model_args)
    print(">Finished starting trainers")
    try:
        for trainer in starters:
            while trainer.is_alive():
                sleep(1)
            #trainer died restart loop
    except KeyboardInterrupt:
        for trainer in starters:
            #trainers will shut down and save, please wait
            while trainer.is_alive():
                sleep(0.1)
            break