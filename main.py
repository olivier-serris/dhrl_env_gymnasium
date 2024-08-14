import envs
import gymnasium as gym


def main():
    env = gym.make("AntMaze", seed=0, render_mode="human")
    env.reset()
    for _ in range(10_000):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        env.render()
        if terminated or truncated:
            env.reset()


main()
