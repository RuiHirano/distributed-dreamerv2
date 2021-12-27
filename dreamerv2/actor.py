import numpy as np
import ray

@ray.remote(num_cpus=1)
class Actor:

  def __init__(self, pid, envs, **kwargs):
    self._pid = pid
    self._envs = envs
    self._kwargs = kwargs
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def rollout(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    eps = []
    while step < steps or episode < episodes:
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          eps.append(ep)
          episode += 1
      self._obs = obs
    return eps, self._pid

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
