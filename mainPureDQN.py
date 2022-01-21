import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
import tensorflow.keras.activations as ka
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as km
import tensorflow_probability as tfp

import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3.dqn import DQN as dqn
import wandb

# from div.render import render_agent
from MEMORY import Memory
from METRICS import *
from div.smoother import Smoother
from div.dummyEnv import DummyEnv

from utils import average_list
from rl_algos.DQN import DQN
from rl_algos.REINFORCE import REINFORCE
from games.connect4 import EnvConnect4
from player import RandomPlayer, PlaySamePlayer, HumanPlayer
from heuristic_functions import heuristic_bad

def run(agent, env, episodes, wandb_cb = True, plt_cb = True):
    print("Run starts.")
    
    k = 0
    N = 9999
    n_points = 10
    config = agent.config
    if wandb_cb: run = wandb.init(project="RL", 
                        entity="timotheboulet",
                        config=config
                        )
    if plt_cb:
        logs = dict()
        
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes} :")
        done = False
        obs = env.reset()
        info = {"mask" : None}

        while not done:
            action = agent.act(obs, mask = info["mask"])
            next_obs, reward, done, info = env.step(action)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()
            env.render()
            
            #Feedback
            k+=1
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
                if plt_cb:
                    
                    for key, value in metric.items():
                        if key not in logs:
                            #logs[key] = {"steps": [agent.step], "values_smoothed": Smoother(value, n_points=n_points)}
                            logs[key] = {"steps": [agent.step], "values": [value]}
                        else:
                            logs[key]["steps"].append(agent.step)
                            logs[key]["values"].append(value) 
                            #logs[key]["values_smoothed"].add_new_value(value)  
                        
                    if k % 100 == 0:       
                        for key, value in logs.items():
                            plt.clf()         
                            plt.plot(logs[key]["steps"][-N:], logs[key]["values"][-N:], '-b')
                            #plt.plot(logs[key]["steps"][-N:], logs[key]["values_smoothed"].list[-N:])
                            plt.title(key)
                            plt.savefig(f"figures/{key}")
                            
            #If episode ended.
            if done:
                print(reward)
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()
    print("End of run.")




if __name__ == "__main__":
    
    #ENV
    #env = gym.make("CartPole-v0")
    env = EnvConnect4(num_cols=7, num_rows=6, opponent=PlaySamePlayer())
    #env = DummyEnv()
    
    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS)

    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Actor_Loss, Metric_Critic_Loss, Metric_Critic_Value]

    #ACTOR PI
    actor = tf.keras.models.Sequential([
        kl.Flatten(),
        kl.Dense(64, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(32, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(env.action_space.n, activation='softmax')
    ])
    #CRITIC Q
    action_value = tf.keras.models.Sequential([
        kl.Flatten(),
        kl.Dense(32, activation='relu', kernel_initializer=ki.he_normal()),
        kl.BatchNormalization(),
        kl.Dense(32, activation='relu', kernel_initializer=ki.he_normal()),
        kl.BatchNormalization(),
        kl.Dense(env.action_space.n, activation='linear'),
    ])

    #AGENT
    agent = DQN(memory = memory, action_value=action_value, metrics = metrics)
    #agent = REINFORCE(memory=memory, actor=actor, metrics=metrics)
    
    #RUN
    run(agent, env, episodes=2, wandb_cb = False, plt_cb=True)






