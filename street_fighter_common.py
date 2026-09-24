"""Shared Street Fighter II Retro helpers."""

import time
from pathlib import Path

try:
    import gymnasium as gym
    import stable_retro as retro
    from gymnasium import spaces

    MODERN_API = True
except ImportError:
    import gym
    import retro
    from gym import spaces

    MODERN_API = False


GAME_NAME = (
    "StreetFighterIISpecialChampionEdition-Genesis-v0"
    if MODERN_API
    else "StreetFighterIISpecialChampionEdition-Genesis"
)
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"
CHAMPION_PLAYERS = (
    "Ryu",
    "Guile",
)
RYU_CAMPAIGN_OPPONENTS = (
    "Guile",
)
RYU_CAMPAIGN_STATES = (DEFAULT_STATE,)
ALL_POSSIBLE_STATES = RYU_CAMPAIGN_STATES
MODEL_DIR = Path("Training") / "Models" / "StreetFighter"
LOG_DIR = Path("Training") / "Logs" / "StreetFighter"


def installed_states():
    """Return the Retro states installed for the Street Fighter integration."""
    metadata_path = Path(
        retro.data.get_file_path(
            GAME_NAME,
            "metadata.json",
            retro.data.Integrations.STABLE,
        )
    )
    return sorted(
        path.stem
        for path in metadata_path.parent.glob("*.state")
        if path.stem in ALL_POSSIBLE_STATES
    )


def resolve_training_states(requested_states=None, allow_missing=False):
    """Resolve CLI state arguments to runnable Retro state names.

    Only the Ryu vs. Guile state is supported, and Retro can launch it only
    when the matching .state file is installed.
    """
    installed = set(installed_states())

    if not requested_states:
        return sorted(installed), []

    if len(requested_states) == 1 and requested_states[0] in {"installed", "all-installed"}:
        return sorted(installed), []

    if len(requested_states) == 1 and requested_states[0] in {"all", "all-possible"}:
        states = list(ALL_POSSIBLE_STATES)
    else:
        states = [state for state in requested_states if state in ALL_POSSIBLE_STATES]

    missing = [state for state in states if state not in installed]
    if allow_missing:
        return states, missing

    runnable = [state for state in states if state in installed]
    return runnable, missing


class StreetFighterMatchWrapper(gym.Wrapper):
    """Reward CPU damage and end episodes when a best-of-three match is over."""

    def __init__(self, env, max_episode_steps=6000, shape_reward=True):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.shape_reward = shape_reward
        self.steps = 0
        self.prev = {}

    def reset(self, **kwargs):
        self.steps = 0
        self.prev = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, _raw_reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, _raw_reward, done, info = result
            terminated, truncated = done, False
        self.steps += 1

        info = dict(info)
        reward = self._reward(info) if self.shape_reward else 0.0
        terminal_reason = None

        matches_won = self._match_count(info.get("matches_won", 0))
        enemy_matches_won = self._match_count(info.get("enemy_matches_won", 0))

        if matches_won >= 2:
            done = True
            terminal_reason = "match_won"
            reward += 5000.0
        elif enemy_matches_won >= 2:
            done = True
            terminal_reason = "match_lost"
            reward -= 5000.0
        elif self.steps >= self.max_episode_steps:
            done = True
            terminal_reason = "timeout"
            health = int(info.get("health", 0))
            enemy_health = int(info.get("enemy_health", 0))
            reward += float(health - enemy_health) * 10.0

        if terminal_reason is not None:
            info["terminal_reason"] = terminal_reason

        self.prev = {
            "health": int(info.get("health", 0)),
            "enemy_health": int(info.get("enemy_health", 0)),
            "matches_won": matches_won,
            "enemy_matches_won": enemy_matches_won,
            "score": int(info.get("score", 0)),
        }
        if len(result) == 5:
            if terminal_reason in {"match_won", "match_lost"}:
                terminated = True
            elif terminal_reason == "timeout":
                truncated = True
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    @staticmethod
    def _match_count(value):
        value = int(value)
        return value if 0 <= value <= 2 else 0

    def _reward(self, info):
        if not self.prev:
            return 0.0

        health = int(info.get("health", self.prev["health"]))
        enemy_health = int(info.get("enemy_health", self.prev["enemy_health"]))
        matches_won = self._match_count(info.get("matches_won", self.prev["matches_won"]))
        enemy_matches_won = self._match_count(info.get("enemy_matches_won", self.prev["enemy_matches_won"]))
        score = int(info.get("score", self.prev["score"]))

        enemy_damage = max(0, self.prev["enemy_health"] - enemy_health)
        self_damage = max(0, self.prev["health"] - health)
        score_delta = max(0, score - self.prev["score"])

        reward = 0.0
        reward += enemy_damage * 5.0
        reward -= self_damage * 3.0
        reward += score_delta * 0.01
        reward += max(0, matches_won - self.prev["matches_won"]) * 1000.0
        reward -= max(0, enemy_matches_won - self.prev["enemy_matches_won"]) * 1000.0
        reward -= 0.01
        return reward


class StreetFighterMacroActionWrapper(gym.Wrapper):
    """Expose useful Ryu move macros as a compact discrete action space."""

    BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

    def __init__(self, env):
        super().__init__(env)
        idle = self.action()
        down = self.action("DOWN")
        left = self.action("LEFT")
        right = self.action("RIGHT")
        up = self.action("UP")
        up_left = self.action("UP", "LEFT")
        up_right = self.action("UP", "RIGHT")
        down_left = self.action("DOWN", "LEFT")
        down_right = self.action("DOWN", "RIGHT")

        self.macro_actions = [
            ("idle", [idle] * 8),
            ("walk_forward", [right] * 16),
            ("crouch", [down] * 16),
            ("hadouken_x", [down] * 4 + [down_right] * 4 + [self.action("RIGHT", "X")] * 8 + [idle] * 8),
            ("hadouken_y", [down] * 4 + [down_right] * 4 + [self.action("RIGHT", "Y")] * 8 + [idle] * 8),
            ("hadouken_z", [down] * 4 + [down_right] * 4 + [self.action("RIGHT", "Z")] * 8 + [idle] * 8),
            ("jump_in_b", [self.action("UP", "RIGHT")] * 30 + [self.action("RIGHT", "B")] * 20 + [right] * 20),
            ("standing_x", [self.action("X")] * 12),
            ("standing_y", [self.action("Y")] * 12),
            ("standing_z", [self.action("Z")] * 12),
            ("crouch_x", [self.action("DOWN", "X")] * 12),
            ("crouch_y", [self.action("DOWN", "Y")] * 12),
            ("crouch_z", [self.action("DOWN", "Z")] * 12),
            ("walk_backward", [left] * 16),
            ("standing_a", [self.action("A")] * 12),
            ("standing_b", [self.action("B")] * 12),
            ("standing_c", [self.action("C")] * 12),
            ("crouch_a", [self.action("DOWN", "A")] * 12),
            ("crouch_b", [self.action("DOWN", "B")] * 12),
            ("crouch_c", [self.action("DOWN", "C")] * 12),
            ("jump_x", [up_right] * 18 + [self.action("RIGHT", "X")] * 16 + [right] * 12),
            ("jump_y", [up_right] * 18 + [self.action("RIGHT", "Y")] * 16 + [right] * 12),
            ("jump_z", [up_right] * 18 + [self.action("RIGHT", "Z")] * 16 + [right] * 12),
            ("jump_a", [up_right] * 18 + [self.action("RIGHT", "A")] * 16 + [right] * 12),
            ("jump_b", [up_right] * 18 + [self.action("RIGHT", "B")] * 16 + [right] * 12),
            ("jump_c", [up_right] * 18 + [self.action("RIGHT", "C")] * 16 + [right] * 12),
            ("shoryuken_x", [right] * 3 + [down] * 3 + [self.action("DOWN", "RIGHT", "X")] * 8 + [idle] * 12),
            ("shoryuken_y", [right] * 3 + [down] * 3 + [self.action("DOWN", "RIGHT", "Y")] * 8 + [idle] * 12),
            ("shoryuken_z", [right] * 3 + [down] * 3 + [self.action("DOWN", "RIGHT", "Z")] * 8 + [idle] * 12),
            ("tatsumaki_a", [down] * 4 + [down_left] * 4 + [self.action("LEFT", "A")] * 10 + [idle] * 12),
            ("tatsumaki_b", [down] * 4 + [down_left] * 4 + [self.action("LEFT", "B")] * 10 + [idle] * 12),
            ("tatsumaki_c", [down] * 4 + [down_left] * 4 + [self.action("LEFT", "C")] * 10 + [idle] * 12),
            ("block_standing", [left] * 20),
            ("block_crouching", [down_left] * 20),
            ("neutral_jump", [up] * 18 + [idle] * 28),
            ("neutral_jump_x", [up] * 18 + [self.action("X")] * 16 + [idle] * 12),
            ("neutral_jump_y", [up] * 18 + [self.action("Y")] * 16 + [idle] * 12),
            ("neutral_jump_z", [up] * 18 + [self.action("Z")] * 16 + [idle] * 12),
            ("neutral_jump_a", [up] * 18 + [self.action("A")] * 16 + [idle] * 12),
            ("neutral_jump_b", [up] * 18 + [self.action("B")] * 16 + [idle] * 12),
            ("neutral_jump_c", [up] * 18 + [self.action("C")] * 16 + [idle] * 12),
            ("back_jump_x", [up_left] * 18 + [self.action("LEFT", "X")] * 16 + [left] * 12),
            ("back_jump_y", [up_left] * 18 + [self.action("LEFT", "Y")] * 16 + [left] * 12),
            ("back_jump_z", [up_left] * 18 + [self.action("LEFT", "Z")] * 16 + [left] * 12),
            ("back_jump_a", [up_left] * 18 + [self.action("LEFT", "A")] * 16 + [left] * 12),
            ("back_jump_b", [up_left] * 18 + [self.action("LEFT", "B")] * 16 + [left] * 12),
            ("back_jump_c", [up_left] * 18 + [self.action("LEFT", "C")] * 16 + [left] * 12),
            ("throw_y", [right] * 20 + [self.action("RIGHT", "Y")] * 12 + [idle] * 8),
            ("throw_z", [right] * 20 + [self.action("RIGHT", "Z")] * 12 + [idle] * 8),
        ]
        self.action_space = spaces.Discrete(len(self.macro_actions))
        self.render_fps = None
        self.render_every_frames = 1
        self.render_frame_count = 0
        self.next_render_time = None
        self.frame_sink = None

    def reset(self, **kwargs):
        self.render_frame_count = 0
        self.next_render_time = None
        return self.env.reset(**kwargs)

    def configure_realtime_render(self, fps, frame_sink=None):
        """Render internal Genesis frames at a paced 30 or 60 FPS."""
        if fps not in (30, 60):
            raise ValueError("Macro rendering supports only 30 or 60 FPS.")
        self.render_fps = fps
        self.render_every_frames = 60 // fps
        self.render_frame_count = 0
        self.next_render_time = None
        self.frame_sink = frame_sink

    def _render_internal_frame(self):
        self.render_frame_count += 1
        if self.render_fps is None or self.render_frame_count % self.render_every_frames:
            return

        now = time.perf_counter()
        if self.next_render_time is None or now - self.next_render_time > 0.25:
            self.next_render_time = now
        elif now < self.next_render_time:
            time.sleep(self.next_render_time - now)

        frame = self.env.render()
        if self.frame_sink is not None and frame is not None:
            self.frame_sink(frame)
        self.next_render_time += 1.0 / self.render_fps

    @classmethod
    def action(cls, *names):
        button_index = {button: index for index, button in enumerate(cls.BUTTONS)}
        action = [0] * len(cls.BUTTONS)
        for name in names:
            action[button_index[name]] = 1
        return action

    def step(self, action):
        _name, sequence = self.macro_actions[int(action)]
        total_reward = 0.0
        done = False
        info = {}
        obs = None

        for low_level_action in sequence:
            result = self.env.step(low_level_action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            total_reward += reward
            self._render_internal_frame()
            if done:
                break

        if len(result) == 5:
            return obs, total_reward, terminated, truncated, info
        return obs, total_reward, done, info


ACTION_MODES = {
    "all": retro.Actions.ALL,
    "discrete": retro.Actions.DISCRETE,
    "filtered": retro.Actions.FILTERED,
    "multi_discrete": retro.Actions.MULTI_DISCRETE,
}


def make_sf2_env(
    state=DEFAULT_STATE,
    seed=None,
    max_episode_steps=6000,
    shape_reward=True,
    action_mode="all",
    macro_actions=False,
    render_mode=None,
):
    if state not in ALL_POSSIBLE_STATES:
        raise ValueError(
            f"Unsupported matchup: {state}. This project only supports {DEFAULT_STATE}."
        )

    if macro_actions:
        action_mode = "all"

    make_kwargs = dict(
        game=GAME_NAME,
        state=state,
        use_restricted_actions=ACTION_MODES[action_mode],
    )
    if MODERN_API:
        make_kwargs["render_mode"] = render_mode or "rgb_array"
    env = retro.make(**make_kwargs)
    if seed is not None:
        if MODERN_API:
            env.action_space.seed(seed)
            env.reset(seed=seed)
        else:
            env.seed(seed)
    env = StreetFighterMatchWrapper(
        env,
        max_episode_steps=max_episode_steps,
        shape_reward=shape_reward,
    )
    if macro_actions:
        env = StreetFighterMacroActionWrapper(env)
    return env


def unpack_reset(result):
    """Return only the observation across Gym and Gymnasium reset APIs."""
    return result[0] if MODERN_API else result


def unpack_step(result):
    """Normalize Gymnasium's five-value step result to the legacy four values."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


def resolve_model_path(model_arg):
    if model_arg and model_arg != "auto":
        return Path(model_arg)

    candidates = [
        MODEL_DIR / "SF2_latest.zip",
        MODEL_DIR / "SF2_final.zip",
        Path("street_fighter_model.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
