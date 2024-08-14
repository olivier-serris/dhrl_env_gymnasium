from gymnasium.envs.registration import registry, register, make, spec

import gymnasium as gym
from envs.antenv import EnvWithGoal, GatherEnv
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.create_gather_env import create_gather_env


def create_antmaze(env_name, seed=None, **kwargs):
    if env_name == "AntGather":
        return GatherEnv(create_gather_env(env_name, seed), env_name)
    elif env_name in [
        "AntMaze",
        "AntMazeSmall-v0",
        "AntMazeComplex-v0",
        "AntMazeSparse",
        "AntPush",
        "AntFall",
    ]:
        env = EnvWithGoal(create_maze_env(env_name, seed, **kwargs), env_name)
    else:
        env = gym.make(env_name)

    return env


for env_name in [
    "AntMaze",
    "AntMazeSmall-v0",
    "AntMazeComplex-v0",
    "AntMazeSparse",
    "AntPush",
    "AntFall",
]:
    register(
        id=env_name,
        entry_point="envs:create_antmaze",
        kwargs={"env_name": env_name},
    )


register(
    id="AntMazeBottleneck-v0",
    entry_point="envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEnv",
    max_episode_steps=600,
    reward_threshold=0.0,
)

register(
    id="AntMazeBottleneck-eval-v0",
    entry_point="envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEvalEnv",
    max_episode_steps=600,
    reward_threshold=0.0,
)

register(
    id="Reacher3D-v0",
    entry_point="envs.fetchenv.create_fetch_env:create_fetch_env",
    kwargs={"env_name": "Reacher3D-v0"},
    max_episode_steps=100,
)

register(
    id="Pusher-v0",
    entry_point="envs.fetchenv.create_fetch_env:create_fetch_env",
    kwargs={"env_name": "Pusher-v0"},
    max_episode_steps=100,
)
