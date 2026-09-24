"""Autoplay Street Fighter II with a trained PPO model."""

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from street_fighter_common import (
    DEFAULT_STATE,
    make_sf2_env,
    resolve_model_path,
    unpack_reset,
    unpack_step,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--action-mode", default="all")
    parser.add_argument("--macro-actions", action="store_true")
    parser.add_argument("--episodes", type=int, default=0, help="0 means run until closed.")
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, choices=(30, 60), default=60)
    parser.add_argument("--render-delay", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")

    env = make_sf2_env(
        state=args.state,
        max_episode_steps=args.max_steps,
        shape_reward=False,
        action_mode=args.action_mode,
        macro_actions=args.macro_actions,
    )
    if args.frame_skip > 1 and not args.macro_actions:
        env = MaxAndSkipEnv(env, skip=args.frame_skip)
    if args.render and args.macro_actions:
        env.configure_realtime_render(args.fps)

    model = PPO.load(str(model_path))
    deterministic = not args.stochastic

    print(f"Loaded model: {model_path}")
    print(f"Playing state: {args.state}")

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            obs = unpack_reset(env.reset())
            done = False
            steps = 0
            last_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _reward, done, last_info = unpack_step(env.step(action))
                steps += 1
                if args.render and not args.macro_actions:
                    env.render()
                    delay = args.render_delay if args.render_delay > 0 else 1.0 / args.fps
                    time.sleep(delay)

            episode += 1
            print(
                "episode={episode} steps={steps} result={result} "
                "matches_won={matches_won} enemy_matches_won={enemy_matches_won} "
                "health={health} enemy_health={enemy_health}".format(
                    episode=episode,
                    steps=steps,
                    result=last_info.get("terminal_reason", "done"),
                    matches_won=last_info.get("matches_won"),
                    enemy_matches_won=last_info.get("enemy_matches_won"),
                    health=last_info.get("health"),
                    enemy_health=last_info.get("enemy_health"),
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
