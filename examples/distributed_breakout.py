import gym
import dreamerv2.api2 as dv2
import os
from pathlib import Path

root_path = Path(os.path.dirname(__file__)).joinpath("..").resolve() # /backend path
scripts_dir = "{}/logdir/distributed_breakout".format(str(root_path))
print(scripts_dir)
config = dv2.defaults.update({
    'logdir': scripts_dir,
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

class BreakoutEnv(gym.Wrapper):
    def __init__(self):
        self.env = gym.make("Breakout-v0")
        super().__init__(self.env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

env = BreakoutEnv()
dv2.train(env, config)

#print(env.reset(), env.observation_space)