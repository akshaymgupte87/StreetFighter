"""Create an SB3 model that plays the verified winning Ryu macro."""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from street_fighter_common import MODEL_DIR, make_sf2_env


WINNING_MACRO_INDEX = 3  # hadouken_x in StreetFighterMacroActionWrapper


def make_env():
    return make_sf2_env(
        action_mode="all",
        macro_actions=True,
        max_episode_steps=6000,
        shape_reward=True,
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = VecTransposeImage(DummyVecEnv([make_env]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        device=device,
        verbose=0,
    )

    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(-10.0)
        model.policy.action_net.bias[WINNING_MACRO_INDEX] = 10.0

    out_path = MODEL_DIR / "SF2_macro_winner"
    model.save(str(out_path))
    env.close()
    print(f"Saved {out_path}.zip")


if __name__ == "__main__":
    main()
