import os
import numpy as np
os.sys.path.insert(0, '../gymjsp')
from gymjsp.jsspenv import BasicJsspEnv, GraphJsspEnv, HeuristicJsspEnv, HeuristicGraphJsspEnv, \
    HeuristicAttentionJsspEnv
from stable_baselines3.common.env_checker import check_env


def check_state(state):
    if isinstance(state, dict):
        feature = state['feature']
        A = state['A']
        D = state['D']
        A = A.todense()
        D = D.todense()
        assert np.all(feature <= 1)
        assert np.all(feature >= -1)
        assert np.all(A <= 1)
        assert np.all(A >= 0)
        assert np.all(D <= 1)
        assert np.all(D >= 0)
    else:
        assert np.all(state <= 1)
        assert np.all(state >= -1)


def check_reward(reward, done):
    if done:
        assert reward == 20
    else:
        assert np.all(reward <= 1)
        assert np.all(reward >= -1)


# check gym like
env = BasicJsspEnv(name='ft06')
check_env(env)

env = HeuristicJsspEnv(name='ft06')
check_env(env)

env = HeuristicAttentionJsspEnv(name='ft06')
check_env(env)

env = GraphJsspEnv(name='ft06')
check_env(env)

env = HeuristicGraphJsspEnv(name='ft06')
check_env(env)

# check state and reward 
env = BasicJsspEnv(name='ft06')
state = env.reset()
check_state(state)
for _ in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    check_state(state)
    check_reward(reward, done)

env = GraphJsspEnv(name='ft06')
state = env.reset()
check_state(state)
for _ in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    check_state(state)
    check_reward(reward, done)

env = HeuristicJsspEnv(name='ft06', schedule_cycle=3)
state = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    check_state(state)
    check_reward(reward, done)

env = HeuristicGraphJsspEnv(name='ft06', schedule_cycle=1)
state = env.reset()
check_state(state)
for _ in range(6):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    check_state(state)
    check_reward(reward, done)

env = HeuristicAttentionJsspEnv(name='ft06', schedule_cycle=2)
state = env.reset()
check_state(state)
for _ in range(6):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    check_state(state)
    check_reward(reward, done)

# check seed 
env = HeuristicAttentionJsspEnv(name='ft06', schedule_cycle=2)
env.seed(1)
state1 = env.reset(random_rate=0.5)
action1 = env.action_space.sample()
env.seed(1)
state2 = env.reset(random_rate=0.5)
action2 = env.action_space.sample()

assert np.all(state1 == state2)
assert action1 == action2

# check reset 
env = HeuristicAttentionJsspEnv(name='ft06', schedule_cycle=2)
state1 = env.reset()
state2 = env.reset(random_rate=0.5)
state3 = env.reset()

assert np.any(state1 != state2)
assert np.all(state1 == state3)

# check random reset 
env = BasicJsspEnv(name='ft06')
env.seed(1)
state1 = env.reset()
env.seed(1)
state2 = env.reset(random_rate=0.5, shuffle=True)
env.seed(1)
state3 = env.reset()
env.seed(1)
state4 = env.reset(random_rate=0.5)

assert np.any(state1 != state2)
assert np.all(state1 == state3)
assert np.any(state2 != state4)

# check reward
env = BasicJsspEnv('ft06', reward_type='utilization')
doable_ops = env.get_doable_ops_in_list()
env.step(doable_ops[1])
doable_ops = env.get_doable_ops_in_list()

action1 = doable_ops[0]
reward1 = env.cal_reward(action1, doable_ops=doable_ops)

action2 = env.wait_action
reward2 = env.cal_reward(action2, doable_ops=doable_ops)

action3 = env.action_space.sample()
while action3 in doable_ops:
    action3 = env.action_space.sample()
reward3 = env.cal_reward(action3, doable_ops=doable_ops)

assert reward2 <= reward1
assert reward3 <= reward2
