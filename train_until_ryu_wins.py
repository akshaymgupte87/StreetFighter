"""Keep training Street Fighter II until Ryu beats Guile."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from street_fighter_common import DEFAULT_STATE, LOG_DIR, MODEL_DIR, make_sf2_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--model-prefix", default="SF2_discrete")
    parser.add_argument("--continue-from", default="auto")
    parser.add_argument("--action-mode", default="discrete")
    parser.add_argument("--macro-actions", action="store_true")
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--max-total-timesteps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-every-chunks", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_train_vec_env(args):
    def make_env(rank):
        def _init():
            env = make_sf2_env(
                state=args.state,
                seed=args.seed + rank,
                max_episode_steps=args.max_episode_steps,
                shape_reward=True,
                action_mode=args.action_mode,
                macro_actions=args.macro_actions,
            )
            if args.frame_skip > 1 and not args.macro_actions:
                env = MaxAndSkipEnv(env, skip=args.frame_skip)
            return Monitor(env, str(LOG_DIR / f"{args.model_prefix}_monitor_{rank}"))

        return _init

    env_fns = [make_env(rank) for rank in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    return VecTransposeImage(vec_cls(env_fns))


def load_or_create_model(args, env):
    latest_path = MODEL_DIR / f"{args.model_prefix}_latest.zip"
    continue_path = latest_path if args.continue_from == "auto" else Path(args.continue_from)

    if continue_path.exists():
        print(f"Continuing from {continue_path}", flush=True)
        model = PPO.load(
            str(continue_path),
            env=env,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            clip_range=0.2,
        )
        model._last_obs = None
        return model, False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting new {args.action_mode} PPO model on {device}", flush=True)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        clip_range=0.2,
    )
    return model, True


def evaluate_checkpoint(model_path, args):
    completed = subprocess.run(
        [
            sys.executable,
            "evaluate_street_fighter.py",
            "--model",
            str(model_path),
            "--state",
            args.state,
            "--action-mode",
            args.action_mode,
            *(["--macro-actions"] if args.macro_actions else []),
            "--episodes",
            str(args.eval_episodes),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--frame-skip",
            str(args.frame_skip),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return payload["wins"], payload["results"]


def write_status(status):
    status_path = LOG_DIR / "ryu_until_win_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = make_train_vec_env(args)
    model, reset_num_timesteps = load_or_create_model(args, env)

    latest_path = MODEL_DIR / f"{args.model_prefix}_latest"
    winner_path = MODEL_DIR / f"{args.model_prefix}_winner"
    total = 0
    chunks = 0

    print(
        f"Training until Ryu wins: state={args.state} action_mode={args.action_mode}",
        flush=True,
    )

    try:
        while args.max_total_timesteps == 0 or total < args.max_total_timesteps:
            remaining = (
                args.chunk_timesteps
                if args.max_total_timesteps == 0
                else min(args.chunk_timesteps, args.max_total_timesteps - total)
            )
            if remaining <= 0:
                break

            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=args.model_prefix,
            )
            reset_num_timesteps = False
            total += remaining
            chunks += 1
            model.save(str(latest_path))

            status = {
                "state": args.state,
                "action_mode": args.action_mode,
                "macro_actions": args.macro_actions,
                "total_timesteps_this_run": total,
                "chunks": chunks,
                "latest_model": str(latest_path) + ".zip",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if chunks % args.eval_every_chunks == 0:
                wins, results = evaluate_checkpoint(str(latest_path) + ".zip", args)
                status["eval_wins"] = wins
                status["eval_episodes"] = args.eval_episodes
                status["eval_results"] = results
                print(json.dumps(status, indent=2), flush=True)

                if wins > 0:
                    model.save(str(winner_path))
                    status["winner_model"] = str(winner_path) + ".zip"
                    status["stopped_reason"] = "ryu_won"
                    write_status(status)
                    print(f"Ryu won. Saved winner to {winner_path}.zip", flush=True)
                    return

            write_status(status)

    except KeyboardInterrupt:
        print("Interrupted; saving latest model.", flush=True)
    finally:
        model.save(str(latest_path))
        env.close()
        print(f"Saved latest model to {latest_path}.zip", flush=True)


if __name__ == "__main__":
    main()
