## Gymjsp

Gymjsp is an open source Python library, which uses the OpenAI Gym interface for easily instantiating and interacting with RL environments, and the OR-Library which is a collection of test data sets for a variety of Operations Research (OR) problems(i.e., job shop schedule, portfolio optimisation and so on).  In this Python library, we only use the job shop schedule instances in OR-Library.

## Installation

To install the Gymjsp library, use `pip install gymjsp`.

## Dependency

To use the Gymjsp library normally, you should install these Python libraries:

* gym
* networkx
* plotly

## API

The Gymjsp API's API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "ft06" instance environment:

```python
from gymjsp import BasicJsspEnv
env = BasicJsspEnv('ft06')

# env is created, now we can use it: 
for episode in range(10): 
    obs = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        nobs, reward, done, info = env.step(action)
        
env.render()
```

## Release Notes

There used to be release notes for all the new Gymjsp versions here. New release notes are being moved to [releases page](https://github.com/Yunhui1998/Gymjsp) on GitHub, like most other libraries do.
