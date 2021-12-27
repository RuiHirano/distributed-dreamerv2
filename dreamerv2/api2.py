import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import ray
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common
from .actor import Actor
from .learner import Learner

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def train(env, config, outputs=None):

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  outputs = outputs or [
      common.TerminalOutput(),
      common.JSONLOutput(config.logdir),
      common.TensorBoardOutput(config.logdir),
  ]
  replay = common.Replay(logdir / 'train_episodes', **config.replay)
  step = common.Counter(replay.stats['total_steps'])
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video = common.Every(config.log_every)
  should_expl = common.Until(config.expl_until)

  env = common.GymWrapper(env)
  env = common.ResizeImage(env)
  if hasattr(env.act_space['action'], 'n'):
    env = common.OneHotAction(env)
  else:
    env = common.NormalizeAction(env)
  env = common.TimeLimit(env, config.time_limit)

  actors = [Actor.remote(pid=i, envs=[env]) for i in range(config.num_actors)]
  learner = Learner.remote(config, env.obs_space, env.act_space, step)
  random_policy = lambda *args: learner.policy.remote(*args, mode='random')

  prefill = max(0, config.prefill - replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    wip_actors = [actor.rollout.remote(random_policy, steps=int(prefill/config.num_actors)) for actor in actors]
    results = ray.get(wip_actors)
    for (episodes, total_steps) in results:
      replay.add_episodes(episodes)


  print('Create agent.')
  #agnt = agent.Agent(config, env.obs_space, env.act_space, step)
  dataset = iter(replay.dataset(**config.dataset))
  if (logdir / 'variables.pkl').exists():
    learner.load.remote(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      wip_learner = learner.train.remote(next(dataset))
      (metrics) = ray.get(wip_learner)[0]
      policy = lambda *args: learner.policy.remote(
          *args, mode='explore' if should_expl(step) else 'train')

  def write_log(metrics):
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(ray.get(learner.report.remote(next(dataset))))
    logger.write(fps=True)

  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.1f}.')
    logger.scalar('return', score)
    logger.scalar('length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{key}', ep[key].max(0).mean())
    if should_video(step):
      for key in config.log_keys_video:
        logger.video(f'policy_{key}', ep[key])
    logger.add(replay.stats)
    logger.write()

  wip_learner = learner.train.remote(next(dataset))
  wip_actors = [actor.rollout.remote(policy, steps=config.actor_steps) for actor in actors]
  while replay.stats["total_steps"] < config.steps:
    logger.write()
    # correct experiments
    finished_actors, wip_actors = ray.wait(wip_actors, timeout=0)
    if finished_actors:
      for finished in finished_actors:
        episodes, pid = ray.get(finished[0])
        replay.add_episodes(episodes)
        wip_actors.extend([actors[pid].rollout.remote(policy, steps=config.actor_steps)])
        for ep in episodes:
          per_episode(ep)

    # train learner
    finished_learner, _ = ray.wait([wip_learner], timeout=0)
    if finished_learner:
      (metrics) = ray.get(wip_learner)[0]
      wip_learner = learner.train.remote(next(dataset))
      # write logs
      write_log(metrics)
