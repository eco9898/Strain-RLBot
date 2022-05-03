import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import *
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import CombinedReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward

from advanced_padder import AdvancedObsPadder
from discrete_act import DiscreteAction
from trainer_classes import *

if __name__ == '__main__':  # Required for multiprocessing
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
    num_instances = 14
    n_env = agents_per_match * num_instances
    print(n_env)
    batch_size = int(((400_000//(n_env))*(n_env))/num_instances) #getting the batch size down to something more manageable - 80k in this case at 5 instances, but 25k at 16 instances
    print(batch_size)
    steps = (1_000_000//batch_size)*batch_size #making sure the experience counts line up properly
    print(steps)
    training_interval = 5_000_000
    print(training_interval)
    mmr_save_frequency = 25_000_000
    print(mmr_save_frequency)

    attackRewards = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            LiuDistancePlayerToBallReward(),
            RewardIfTouchedLast(LiuDistanceBallToGoalReward()),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            RewardIfClosestToBall(AlignBallGoal(0,1), True),
        ),
        (1.0, 0.2, 1.0, 1.0, 0.8))

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
                    ),
                    (40.0, 1.0, 2.0, 0.2)
                ),
                team_only=True
            ),
            RewardIfMidFromBall(defendRewards),
            RewardIfFurthestFromBall(lastManRewards),
            TeamSpacingReward(),
            pickupBoost()
        ),
        (2.0, 1.0, 1.5, 1.0, 0.4))

    def exit_save(model):
        model.save("src/models/exit_save")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size, #amount of bots per team
            tick_skip=frame_skip,
            reward_function=CombinedReward(
            (
                RewardIfAttacking(attackRewards),
                RewardIfDefending(defendRewards),
                RewardIfLastMan(lastManRewards),
                RewardIfKickoff(kickoffRewards),
                VelocityReward(),
                FaceBallReward(),
                SaveBoostReward(),
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
                TeamSpacingReward(),
                FlipReward(),
                pickupBoost()
            ),
            (1.0, 1.0, 3.0, 5.0, 1.0, 1.0, 2.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.4)),
            self_play=self_play, #play against its self
            #time out after 50 seconds encourage kickoff
            terminal_conditions=[TimeoutCondition(fps * 20), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition(), BallTouchedCondition()],
            obs_builder=AdvancedObsPadder(3),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    paging = True
    wait_time=22
    if paging:
        wait_time=40

    env = SB3MultipleInstanceEnv(get_match, num_instances, force_paging=paging, wait_time=wait_time) #or 40            # Start instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
            "src/models/exit_save",
            env,
            device="auto",
            #custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
            #If you need to adjust parameters mid training, you can use the below example as a guide
            custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "_last_obs": None}
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

        model = PPO(
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
            tensorboard_log="src/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(1_000_000 / env.num_envs), save_path="src/models", name_prefix="rl_model") # backup every 5 mins
    #80000 a minute?

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        training_interval = training_interval - (model.num_timesteps % training_interval)
        while True:
            print("training for %s timesteps" % training_interval)
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
            exit_save(model)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"src/mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")