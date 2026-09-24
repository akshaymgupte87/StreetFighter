"""Run Street Fighter II Special Champion Edition through Gym Retro."""

import argparse
import time

from street_fighter_common import (
    DEFAULT_STATE,
    GAME_NAME,
    make_sf2_env,
    unpack_reset,
    unpack_step,
)

GAME = GAME_NAME
STATE = DEFAULT_STATE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        env = make_sf2_env(
            state=STATE,
            shape_reward=False,
            render_mode="rgb_array",
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Import your legally owned Street Fighter II Genesis ROM first, for example:\n"
            "wsl -d Ubuntu -- /home/akshay/.venvs/streetfighter314/bin/python "
            "-m retro.import /mnt/c/path/to/rom_folder"
        ) from exc

    obs = unpack_reset(env.reset())
    print(f"Started {GAME} / {STATE}; observation shape: {getattr(obs, 'shape', None)}")

    try:
        for _ in range(args.steps):
            obs, reward, done, info = unpack_step(env.step(env.action_space.sample()))
            if args.render:
                env.render()
                if args.render_delay > 0:
                    time.sleep(args.render_delay)
            if done:
                obs = unpack_reset(env.reset())
    finally:
        env.close()


if __name__ == "__main__":
    main()
