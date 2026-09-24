"""Train a PPO agent for Street Fighter II Special Champion Edition."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from street_fighter_common import LOG_DIR, MODEL_DIR, make_sf2_env, resolve_model_path, resolve_training_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument(
        "--states",
        nargs="*",
        default=None,
        help="State to train on. This project supports only Champion.Level1.RyuVsGuile.",
    )
    parser.add_argument(
        "--allow-missing-states",
        action="store_true",
        help="Pass requested state names through even when matching .state files are not installed.",
    )
    parser.add_argument("--action-mode", default="discrete", choices=("all", "discrete", "filtered", "multi_discrete"))
    parser.add_argument("--macro-actions", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-prefix", default="SF2_discrete")
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--continue-from", default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def make_training_env(state, rank, args):
    def _init():
        env = make_sf2_env(
            state=state,
            seed=args.seed + rank,
            max_episode_steps=args.max_episode_steps,
            shape_reward=True,
            action_mode=args.action_mode,
            macro_actions=args.macro_actions,
        )
        if args.frame_skip > 1 and not args.macro_actions:
            env = MaxAndSkipEnv(env, skip=args.frame_skip)
        monitor_name = LOG_DIR / f"monitor_{rank}_{state}"
        env = Monitor(env, str(monitor_name))
        return env

    return _init


def build_vec_env(states, args):
    env_fns = [
        make_training_env(states[index % len(states)], index, args)
        for index in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)
    return VecTransposeImage(env)


def main():
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    states, missing_states = resolve_training_states(
        args.states,
        allow_missing=args.allow_missing_states,
    )
    if not states:
        raise SystemExit(
            "No runnable Street Fighter Retro states are installed for the requested selection. "
            "Create or import matching .state files first."
        )

    if missing_states:
        print(
            f"Skipping {len(missing_states)} requested states without installed .state files. "
            "Use --allow-missing-states only if you want Retro to try them anyway."
        )

    print("Training states:")
    for state in states:
        print(f"  {state}")

    env = build_vec_env(states, args)
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(MODEL_DIR),
        name_prefix=f"{args.model_prefix}_checkpoint",
        verbose=1,
    )

    continue_path = (
        MODEL_DIR / f"{args.model_prefix}_latest.zip"
        if args.continue_from == "auto"
        else resolve_model_path(args.continue_from)
    )
    if continue_path.exists():
        print(f"Continuing from {continue_path}")
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
        reset_num_timesteps = True
    else:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Starting a new PPO model on {device} with action_mode={args.action_mode}")
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
        reset_num_timesteps = True

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint,
            log_interval=args.log_interval,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving latest checkpoint.")
    finally:
        latest_path = MODEL_DIR / f"{args.model_prefix}_latest"
        final_path = MODEL_DIR / f"{args.model_prefix}_final"
        model.save(str(latest_path))
        model.save(str(final_path))
        env.close()
        print(f"Saved latest model to {latest_path}.zip")
        print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
