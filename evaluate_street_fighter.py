"""Evaluate a Street Fighter II model and print JSON results."""

import argparse
import json

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from street_fighter_common import DEFAULT_STATE, make_sf2_env, unpack_reset, unpack_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--action-mode", default="discrete")
    parser.add_argument("--macro-actions", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--frame-skip", type=int, default=4)
    return parser.parse_args()


def run_episode(model, args, episode):
    env = make_sf2_env(
        state=args.state,
        max_episode_steps=args.max_episode_steps,
        shape_reward=False,
        action_mode=args.action_mode,
        macro_actions=args.macro_actions,
    )
    if args.frame_skip > 1 and not args.macro_actions:
        env = MaxAndSkipEnv(env, skip=args.frame_skip)

    obs = unpack_reset(env.reset())
    done = False
    steps = 0
    info = {}

    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, info = unpack_step(env.step(action))
            steps += 1
    finally:
        env.close()

    won = int(info.get("matches_won", 0)) >= 2
    return {
        "episode": episode,
        "won": won,
        "steps": steps,
        "terminal_reason": info.get("terminal_reason", "done"),
        "matches_won": int(info.get("matches_won", 0)),
        "enemy_matches_won": int(info.get("enemy_matches_won", 0)),
        "health": int(info.get("health", 0)),
        "enemy_health": int(info.get("enemy_health", 0)),
    }


def main():
    args = parse_args()
    model = PPO.load(args.model, device="cpu")
    results = [run_episode(model, args, episode) for episode in range(1, args.episodes + 1)]
    payload = {
        "wins": sum(1 for result in results if result["won"]),
        "episodes": args.episodes,
        "results": results,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
