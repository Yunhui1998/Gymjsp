import os
os.sys.path.insert(0, '../gymjsp')
from gymjsp.jsspenv import HeuristicAttentionJsspEnv

env = HeuristicAttentionJsspEnv('swv01', schedule_cycle=10)
env.reset()

for _ in range(300):
    env.step(env.action_space.sample())  # take a random action

env.render()
