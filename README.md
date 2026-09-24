# Street Fighter AI: Ryu vs. Guile

A reinforcement learning and scripted gameplay project for *Street Fighter II': Special Champion Edition* on Sega Genesis. Stable Retro provides emulation, and Stable-Baselines3 PPO trains a CNN policy from game frames.

The supported matchup is `Champion.Level1.RyuVsGuile`. The project includes random-input playback, a scripted Ryu demo, a 48-move notebook showcase, and model training and evaluation.

## Setup

The primary configuration targets Python 3.14 in Ubuntu on WSL2. `requirements.txt` pins Stable Retro, Gymnasium, Stable-Baselines3, PyTorch, and notebook dependencies. The older native Windows configuration is retained in `requirements-legacy-windows.txt`.

From PowerShell, enter Ubuntu:

```powershell
wsl -d Ubuntu
```

With Python 3.14 installed, run these commands in the **Ubuntu shell**. Adjust the checkout path as needed:

```bash
cd /mnt/c/Users/aksha/PycharmProjects/StreetFighter
python3.14 -m venv ~/.venvs/streetfighter314
source ~/.venvs/streetfighter314/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorboard
```

TensorBoard is needed by the training scripts' logging configuration but is not explicitly listed in `requirements.txt`. All following Python commands assume this activated environment and the project directory as the working directory. Separate game windows require a graphical display, such as WSLg.

Place your own game ROM in `roms/` and import it:

```bash
mkdir -p roms
python -m retro.import ./roms
```

The Retro integration must also contain the `Champion.Level1.RyuVsGuile` save state. Importing a ROM does not create missing save states.

ROMs, generated diagnostics, and model checkpoints are excluded from Git. Train a model or create the deterministic macro model below before using saved-model playback.

## Quick start

Run random controller input, or watch the predefined Ryu move sequence:

```bash
python run_street_fighter.py --steps 300
python run_street_fighter.py --render --steps 10000
python watch_ryu_beat_guile.py --episodes 1 --delay 0.03
```

The watch script repeats a predefined sequence; it does not use a learned PPO policy. Press `Ctrl+C` to stop playback.

## Notebook showcase

`street_fighter_mixed_moves.ipynb` showcases 48 moves, including movement, blocks, throws, normal attacks, and special moves, with mixed combinations. Set `FPS = 30` or `FPS = 60` in its cell to control playback speed.

```bash
python -m ipykernel install --user --name streetfighter314 --display-name "StreetFighter Python 3.14 (WSL2)"
python -m notebook street_fighter_mixed_moves.ipynb
```

Select the registered kernel before running the cell. The Windows launcher `open_street_fighter_notebook.cmd` is also available, but contains machine-specific paths to the checkout and `/home/akshay/.venvs/streetfighter314`. Update those paths for another machine.

## Train a PPO agent

```bash
python train_street_fighter.py --total-timesteps 2000000 --action-mode discrete --model-prefix SF2_discrete
```

Training uses shaped rewards and saves checkpoints in `Training/Models/StreetFighter/`, with logs in `Training/Logs/StreetFighter/`. It automatically resumes from `<model-prefix>_latest.zip` if present. Use a new prefix for a separate run, or `--continue-from PATH` for a specific compatible checkpoint. Interrupting training with `Ctrl+C` saves latest and final models.

Useful options include `--device cpu`, `--n-envs 4 --subproc`, and `--macro-actions`. Macro actions execute predefined controller sequences. Keep the action configuration consistent when resuming, playing, or evaluating a model.

Alternate training and evaluation until an evaluation episode is won, with a timestep budget:

```bash
python train_until_ryu_wins.py --chunk-timesteps 50000 --max-total-timesteps 2000000 --eval-episodes 3
```

Without `--max-total-timesteps`, this script has no training limit. Winning is not guaranteed. Progress is written to `Training/Logs/StreetFighter/ryu_until_win_status.json`; a successful evaluation saves `<model-prefix>_winner.zip`.

## Play and evaluate

Pass an explicit checkpoint and match the training action mode:

```bash
python play_street_fighter.py --model Training/Models/StreetFighter/SF2_discrete_latest.zip --action-mode discrete --render --episodes 3 --fps 60
python evaluate_street_fighter.py --model Training/Models/StreetFighter/SF2_discrete_latest.zip --action-mode discrete --episodes 3
```

Playback runs indefinitely when `--episodes` is omitted. Evaluation prints JSON containing total wins and per-episode results, including health, round wins, and termination reasons. An episode counts as a win when Ryu has won at least two rounds.

Playback's `--model auto` searches for `SF2_latest.zip`, then `SF2_final.zip` in the model directory, then `street_fighter_model.zip` in the project root. It does not automatically select `SF2_discrete_latest.zip`.

### Deterministic macro model

Create a PPO-format model manually configured to select the Hadouken macro:

```bash
python create_street_fighter_macro_model.py
python play_street_fighter.py --model Training/Models/StreetFighter/SF2_macro_winner.zip --macro-actions --render --episodes 1
python evaluate_street_fighter.py --model Training/Models/StreetFighter/SF2_macro_winner.zip --macro-actions --episodes 3
```

This model is constructed directly rather than learned through training. Creating it overwrites `SF2_macro_winner.zip` if that file already exists.

## Project files

| File | Purpose |
| --- | --- |
| `street_fighter_common.py` | Environment creation, rewards, macro actions, and Gym/Gymnasium compatibility |
| `run_street_fighter.py` | Random-input environment smoke run |
| `watch_ryu_beat_guile.py` | Scripted sequence playback |
| `street_fighter_mixed_moves.ipynb` | Inline move showcase and combinations |
| `train_street_fighter.py` | PPO training and checkpoint saving |
| `train_until_ryu_wins.py` | Repeated training and evaluation |
| `play_street_fighter.py` | Saved-model playback |
| `evaluate_street_fighter.py` | Evaluation with JSON output |
| `create_street_fighter_macro_model.py` | Deterministic macro-model generation |
| `Training/` | Model checkpoints and logs |

## Troubleshooting

- **ROM or state not found:** import the ROM using the same environment that runs the scripts, and check that the supported save state is installed.
- **Missing modules:** activate the WSL virtual environment and install `requirements.txt`. Install `tensorboard` for training.
- **Action-space mismatch:** match `--action-mode` and `--macro-actions` to the checkpoint. Playback defaults to `all`; training and evaluation default to `discrete`.
- **No game window:** check the WSL graphical display, or use the notebook's inline frame display.
- **Model not found:** pass an explicit model path and run commands from the project root.
