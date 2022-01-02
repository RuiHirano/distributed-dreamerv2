import collections
import logging
import os
import pathlib
import re
import sys
import time
import warnings
import ray
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
import numpy as np
import ruamel.yaml as yaml

#from actor import Actor
from learner_actor import Learner, Actor
import common

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def train(env, config, outputs=None):
  ray.init()
  def add_path(*args, **kwargs):
    sys.path.append(str(pathlib.Path(__file__).parent))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger().setLevel('ERROR')
    warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
  ray.worker.global_worker.run_function_on_all_workers(add_path)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  outputs = outputs or [
      #common.TerminalOutput(),
      common.JSONLOutput(config.logdir),
      common.TensorBoardOutput(config.logdir),
  ]
  replay = common.Replay(logdir / 'train_episodes', **config.replay)
  step = common.Counter(replay.stats['total_steps'])
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_video = common.Every(config.log_every)
  should_expl = common.Until(config.expl_until)
  should_log = common.Every(config.log_every)

  env = common.GymWrapper(env)
  env = common.ResizeImage(env)
  if hasattr(env.act_space['action'], 'n'):
    env = common.OneHotAction(env)
  else:
    env = common.NormalizeAction(env)
  env = common.TimeLimit(env, config.time_limit)
  actors = [Actor.remote(pid=i, env=env, config=config) for i in range(config.num_actors)]
  learner = Learner.remote(config, env)
  variables = ray.get(learner.get_variables.remote())
  variables = ray.put(variables)
  #random_policy = lambda *args: learner.policy.remote(*args, mode='random')
  #random_policy = ray.put(random_policy)

  prefill = max(0, config.prefill - replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    wip_actors = [actor.rollout.remote(variables, steps=int(prefill/config.num_actors), mode='random') for actor in actors]
    results = ray.get(wip_actors)
    for episodes, _ in results:
      replay.add_episodes(episodes)

  print('Create agent.')
  #agnt = agent.Agent(config, env.obs_space, env.act_space, step)
  dataset = iter(replay.dataset(**config.dataset))
  if (logdir / 'variables.pkl').exists():
    learner.load.remote(logdir / 'variables.pkl')
    variables = ray.get(learner.get_variables.remote())
    variables = ray.put(variables)
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      variables, mets, info = ray.get(learner.train.remote(next(dataset)))
      variables = ray.put(variables)
  #policy = lambda *args: learner.policy.remote(
  #    *args, mode='explore' if should_expl(step) else 'train')

  def write_log(metrics):
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(ray.get(learner.report.remote(next(dataset))))
    logger.write(fps=True)

  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
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
    return length, score

  wip_learner = learner.train.remote(next(dataset))
  wip_actors = [actor.rollout.remote(variables, steps=config.actor_steps, mode='train') for actor in actors]
  actor_cycle = 0
  train_num = 0
  prev_total_steps = replay.stats["total_steps"]
  while replay.stats["total_steps"] < config.steps:
    logger.write()
    # correct experiments
    finished_actors, wip_actors = ray.wait(wip_actors, timeout=0)
    if finished_actors:
      for finished in finished_actors:
        actor_cycle += 1
        episodes, info = ray.get(finished)
        replay.add_episodes(episodes)
        if should_expl(step):
          wip_actors.extend([actors[info["pid"]].rollout.remote(variables, steps=config.actor_steps, mode='explore')])
        # log
        mean_length, mean_score = 0, 0
        for ep in episodes:
          length, score = per_episode(ep)
          mean_length += length
          mean_score += score
        mean_length, mean_score = mean_length/len(episodes), mean_score/len(episodes)
        print("[{}] Actor Cycle: {}, Steps, {}, PID: {}, Episode has {:.0f} steps and return {:.1f}., time: {}".format(replay.stats["total_steps"], actor_cycle, config.actor_steps, info["pid"], mean_length, mean_score, info["elapsed_time"]))

    # train learner
    finished_learner, _ = ray.wait([wip_learner], timeout=0)
    if finished_learner:
      variables, mets, info = ray.get(finished_learner[0])
      [metrics[key].append(value) for key, value in mets.items()]
      variables = ray.put(variables)
      train_num += 1
      print("Finished learner train: {}, actor_cycle: {}, steps: {}, time: {}".format(train_num, actor_cycle, replay.stats["total_steps"]-prev_total_steps, info["elapsed_time"]))
      actor_cycle = 0
      # write logs
      if should_log(step):
        write_log(metrics)
      wip_learner = learner.train.remote(next(dataset))
