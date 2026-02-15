"""
Watch a trained JAX Jamb agent play a game.
Same output format as watch_agent.py (v4/v5/v6).
Can be launched from Windows PowerShell (auto-relays to WSL2).
"""

import os
import sys
import platform
import subprocess

# ── Auto-relay to WSL2 if running on Windows ─────────────────────────
def _relay_to_wsl():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    drive = script_dir[0].lower()
    wsl_dir = f'/mnt/{drive}/' + script_dir[3:].replace('\\', '/')
    wsl_script = f'{wsl_dir}/{os.path.basename(__file__)}'
    # Pass through all CLI arguments
    extra_args = ' '.join(f'"{a}"' if ' ' in a else a for a in sys.argv[1:])
    cmd = [
        'wsl', '-d', 'Ubuntu-22.04', '-u', 'root', '--',
        'bash', '-c',
        f'cd "{wsl_dir}" && python3 "{wsl_script}" {extra_args}'
    ]
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        sys.exit(0)

if platform.system() == 'Windows':
    _relay_to_wsl()

# ── Imports (only reached inside WSL2) ───────────────────────────────
# Force CPU-only — watch script doesn't need GPU and training may be using it
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import jamb_jax as env


# ─── Network (must match train_jax.py) ───────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int
    actor_layers: list
    critic_layers: list
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        actor = x
        for size in self.actor_layers:
            actor = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(actor)
            actor = act_fn(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0))(actor)
        critic = x
        for size in self.critic_layers:
            critic = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(critic)
            critic = act_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                        bias_init=constant(0.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


# ─── Display Helpers (same format as watch_agent.py) ──────────────────

ROWS = ["1s", "2s", "3s", "4s", "5s", "6s", "Max", "Min", "T", "K", "F", "P", "Y"]
COLS = ["Down", "Free", "Up", "Anno"]


def render_board(board):
    """Render the game board. board is (13,4) numpy array, -1 = empty."""
    header = "      " + " ".join([f"{c:^8}" for c in COLS])
    print("\n" + header)
    print("      " + "-" * 36)
    for r_idx, row_name in enumerate(ROWS):
        line = f"{row_name:>4} | "
        for c_idx in range(4):
            val = int(board[r_idx, c_idx])
            display = "." if val == -1 else str(val)
            line += f"{display:^8} "
        print(line)
    print("      " + "-" * 36)


def format_histogram(hist):
    """Format histogram as human-readable dice list, e.g. '[1, 1, 5, 6, 6]'."""
    dice = []
    for val_idx in range(6):
        count = int(hist[val_idx])
        dice.extend([val_idx + 1] * count)
    return str(dice)


def format_keep_pattern(pattern):
    """Format a keep pattern as a list of dice."""
    kept_dice = []
    for val_idx in range(6):
        count = int(pattern[val_idx])
        kept_dice.extend([val_idx + 1] * count)
    return str(kept_dice) if kept_dice else "nothing"


# ─── Load Model ──────────────────────────────────────────────────────

def load_model(model_path):
    """Load a saved JAX model checkpoint."""
    network = ActorCritic(
        action_dim=env.TOTAL_ACTIONS,
        actor_layers=[512, 512, 256],
        critic_layers=[512, 512, 256],
        activation="relu",
    )

    # Init network to get tree structure
    dummy_obs = jnp.zeros(env.OBS_SIZE)
    rng = jax.random.PRNGKey(0)
    params = network.init(rng, dummy_obs)

    # Load saved params
    data = np.load(model_path)
    param_leaves = [data[f"param_{i}"] for i in range(len(data.files))]
    tree_struct = jax.tree_util.tree_structure(params)
    loaded_params = jax.tree_util.tree_unflatten(tree_struct, param_leaves)

    return network, loaded_params


# ─── Watch Agent ──────────────────────────────────────────────────────

def watch_agent(model_path, delay=0.5):
    """Watch the agent play one complete game."""
    try:
        network, params = load_model(model_path)
        print(f"\n🚀 LOADED AGENT: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # JIT the forward pass
    @jax.jit
    def predict(params, obs, mask):
        logits, value = network.apply(params, obs)
        masked_logits = jnp.where(mask, logits, -1e8)
        return jnp.argmax(masked_logits), value

    # Reset environment
    key = jax.random.PRNGKey(int(time.time()))
    state, obs = env.reset(key)
    last_turn_printed = -1

    print("\n" + "=" * 40)
    print("      JAMB RL AGENT - LIVE GAME (JAX)")
    print("=" * 40)

    while not bool(state.game_over):
        turn_now = int(state.turn_number)
        rolls_before = int(state.rolls_left)
        hist_before = np.array(state.dice_hist)

        if turn_now != last_turn_printed:
            print(f"\n--- TURN {turn_now} ---")
            last_turn_printed = turn_now

        print(f"🎲 Dice: [{format_histogram(hist_before)}] (Rolls Remaining: {rolls_before})")

        mask = env.get_action_mask(state)
        action, value = predict(params, obs, mask)
        action = int(action)

        # Execute action
        key, step_key = jax.random.split(key)
        state, obs, reward, done, info = env.step(step_key, state, jnp.int32(action))

        if action < env.NUM_KEEP_ACTIONS:
            pattern = np.array(env.KEEP_PATTERNS[action])
            print(f"👉 ACTION: ROLL (Kept: {format_keep_pattern(pattern)})")
        elif action < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS:
            idx = action - env.NUM_KEEP_ACTIONS
            r_idx, c_idx = idx // 4, idx % 4
            print(f"👉 ACTION: SCORE {ROWS[r_idx]} in {COLS[c_idx]}")
            render_board(np.array(state.board))
            total = int(env.calculate_total_score(state.board))
            print(f"🏆 Total Score: {total}")
            print("-" * 40)
        else:
            row = action - env.NUM_KEEP_ACTIONS - env.NUM_SCORE_ACTIONS
            print(f"👉 ACTION: ANNOUNCE {ROWS[row]}")

        if delay > 0:
            time.sleep(delay)

    print(f"\n🏁 GAME OVER!")
    total = int(env.calculate_total_score(state.board))
    print(f"✨ Final Score: {total}")


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Auto-detect latest model
    latest_model = None
    # Prioritize V2, then V1
    for version_dir in ["models/jamb_jax_v2", "models/jamb_jax_v1"]:
        if os.path.exists(version_dir):
            files = [f for f in os.listdir(version_dir)
                     if f.endswith(".npz") and not f.endswith("_tree.json")]
            if files:
                def get_step_count(fname):
                    if "final" in fname:
                        return float('inf')
                    try:
                        # ckpt_123456.npz
                        parts = fname.replace(".npz", "").split("_")
                        return int(parts[-1])
                    except ValueError:
                        return -1
                
                # Sort by step count descending
                files.sort(key=get_step_count, reverse=True)
                latest_model = os.path.join(version_dir, files[0])
                # If we found a model in V2, stop looking
                break

    if latest_model is None:
        latest_model = "models/jamb_jax_v1/jamb_jax_v1_final.npz"

    parser.add_argument("--model", type=str, default=latest_model, help="Path to .npz checkpoint")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps (seconds)")
    args = parser.parse_args()

    watch_agent(args.model, args.delay)
