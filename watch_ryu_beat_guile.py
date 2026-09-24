"""Render the verified winning Ryu sequence against Guile."""

import argparse
import time

from street_fighter_common import make_sf2_env, unpack_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_sf2_env(
        action_mode="all",
        macro_actions=True,
        shape_reward=False,
        render_mode="rgb_array",
    )
    _name, sequence = env.macro_actions[3]

    try:
        for episode in range(1, args.episodes + 1):
            env.reset()
            info = {}
            done = False
            steps = 0

            while not done and int(info.get("matches_won", 0)) < 2:
                for action in sequence:
                    _obs, _reward, done, info = unpack_step(env.env.step(action))
                    env.render()
                    time.sleep(args.delay)
                    steps += 1
                    if done or int(info.get("matches_won", 0)) >= 2:
                        break

            print(
                "episode={episode} steps={steps} matches_won={matches_won} "
                "enemy_matches_won={enemy_matches_won} health={health} enemy_health={enemy_health}".format(
                    episode=episode,
                    steps=steps,
                    matches_won=info.get("matches_won"),
                    enemy_matches_won=info.get("enemy_matches_won"),
                    health=info.get("health"),
                    enemy_health=info.get("enemy_health"),
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
