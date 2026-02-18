"""
Evaluate JAX Jamb Agent - 100k Games (Universal)
================================================
Runs 100,000 games on GPU to gather comprehensive statistics.
Automatically adapts to V2 (178 obs) or V3/V4 (230 obs) models.
"""

# ─── Configuration ───────────────────────────────────────────────────
# MODIFY THIS PATH TO THE MODEL YOU WANT TO EVALUATE
MODEL_PATH = "models/jamb_jax_v4/jamb_jax_v1_final.npz"
# ─────────────────────────────────────────────────────────────────────

import os
import sys
import platform
import subprocess
import time
import numpy as np

# ── Auto-relay to WSL2 if running on Windows ─────────────────────────
def _relay_to_wsl():
    """Relaunch this script inside WSL2."""
    script_path = os.path.abspath(__file__)
    # C:\Path\To\File.py -> /mnt/c/Path/To/File.py
    drive = script_path[0].lower()
    wsl_path = f"/mnt/{drive}" + script_path[2:].replace("\\", "/")
    
    print(f"🔄 Relaying to WSL2: {wsl_path}")
    cmd = ["wsl", "-d", "Ubuntu-22.04", "-u", "root", "--", "bash", "-c", 
           f"cd '{os.path.dirname(wsl_path)}' && python3 '{os.path.basename(wsl_path)}'"]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if platform.system() == 'Windows':
    _relay_to_wsl()

# ── Imports (only reached inside WSL2) ───────────────────────────────
import jax
import jax.numpy as jnp
import flax.linen as nn
import jamb_jax as env
from watch_agent_jax import ActorCritic, load_model, ROWS, COLS

TOTAL_GAMES = 1_000_000
BATCH_SIZE = 5_000

# ─── Evaluation Logic ────────────────────────────────────────────────

# ─── Evaluation Logic ────────────────────────────────────────────────

def run_single_game_impl(key, model_params, obs_dim):
    """Run a single game to completion and return stats."""
    # We instantiate the network, but apply() will validate shapes
    network = ActorCritic(
        action_dim=env.TOTAL_ACTIONS,
        actor_layers=[512, 512, 256],
        critic_layers=[512, 512, 256],
        activation="relu",
    )

    def cond_fn(carry):
        state, _, _, _, done = carry
        return ~done

    def body_fn(carry):
        state, _, _, fill_turns, _ = carry
        
        # Get action
        obs_full = env.get_obs(state)
        # Adapt observation to model's expected size
        obs = obs_full[:obs_dim] 
        
        mask = env.get_action_mask(state)
        logits, _ = network.apply(model_params, obs)
        masked_logits = jnp.where(mask, logits, -1e9)
        action = jnp.argmax(masked_logits)
        
        # Step
        k1, k2 = jax.random.split(carry[2]) 
        next_state, _, _, done, _ = env.step(k1, state, action)
        
        # Track column fills
        cols_filled = jnp.all(next_state.board >= 0, axis=0) # (4,) bool
        current_turn = state.turn_number 
        
        new_fill_turns = jnp.where(
            (fill_turns == 100) & (cols_filled),
            current_turn,
            fill_turns
        )
        
        return next_state, 0.0, k2, new_fill_turns, done

    # Initial state
    k_reset, k_loop = jax.random.split(key)
    state, _ = env.reset(k_reset)
    initial_fills = jnp.full((4,), 100, dtype=jnp.int32)
    
    final_state, _, _, final_fills, _ = jax.lax.while_loop(
        cond_fn, body_fn, (state, 0.0, k_loop, initial_fills, False)
    )
    
    final_score = env.calculate_total_score(final_state.board)
    return final_state.board, final_score, final_fills

run_single_game = jax.jit(run_single_game_impl, static_argnums=(2,))

def run_batch_impl(keys, model_params, obs_dim):
    """Run a batch of games using provided keys."""
    # Use partial to fix obs_dim since it's a static integer, not a traced array
    from functools import partial
    fn = partial(run_single_game, obs_dim=obs_dim)
    return jax.vmap(fn, in_axes=(0, None))(keys, model_params)

run_batch = jax.jit(run_batch_impl, static_argnums=(2,))

# ─── Main Analysis ───────────────────────────────────────────────────

def generate_report(stats, best_game_log):
    scores = stats['scores']
    boards = stats['boards']
    fill_turns = stats['fill_turns'] # (N, 4)

    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    
    # Percentiles
    pcts = [1, 10, 25, 50, 75, 90, 99]
    res = np.percentile(scores, pcts)
    
    # Board Stats
    avg_cell_values = np.mean(boards, axis=0) # (13, 4)
    avg_fills = np.mean(fill_turns, axis=0)   # (4,)

    report = f"""# Evaluation Report: {stats['model_name']}
**Games:** {TOTAL_GAMES:,} | **Device:** GPU (WSL2)
**Obs Dim:** {stats['obs_dim']} (Environment: {env.OBS_SIZE})

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **{avg_score:.2f}** |
| **Max** | **{max_score:.0f}** |
| Median | {res[3]:.1f} |
| StdDev | {std_score:.2f} |
| Min | {min_score:.0f} |

### Percentiles
| % | Score |
|---|---|
"""
    for p, v in zip(pcts, res):
        report += f"| {p}% | {v:.0f} |\n"

    report += f"""
## ⏱️ Column Completion Speed
Average turn number when the column was fully filled (Lower is faster).

| Column | Avg Turn Filled |
|:---|:---|
| **Down** | {avg_fills[0]:.1f} |
| **Free** | {avg_fills[1]:.1f} |
| **Up** | {avg_fills[2]:.1f} |
| **Anno** | {avg_fills[3]:.1f} |

## 🎲 Average Board Values
(Averaged across {TOTAL_GAMES:,} games)

| Row | Down | Free | Up | Anno |
|:----|:---:|:---:|:---:|:---:|
"""
    for r, row_name in enumerate(ROWS):
        vals = avg_cell_values[r]
        report += f"| **{row_name}** | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.2f} |\n"

    report += f"""
## 📜 Best Game Log (Score: {max_score})
Seed: `{stats['best_seed']}`

```text
{best_game_log}
```
"""
    return report

def capture_game_log(model_path, seed, obs_dim):
    """Replays the game with the given seed and returns the log string."""
    network, params = load_model(model_path)
    
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    
    with redirect_stdout(f):
        print(f"--- Replaying Game with Seed {seed} ---")
        pk = jax.random.PRNGKey(seed)
        k_reset, k_game = jax.random.split(pk)
        state, _ = env.reset(k_reset)
        
        step_jit = jax.jit(env.step)
        # Sliced prediction
        predict_jit = jax.jit(lambda p, o, m: jnp.argmax(jnp.where(m, network.apply(p, o[:obs_dim])[0], -1e9)))
        
        rng = k_game
        from watch_agent_jax import format_histogram, format_keep_pattern, render_board
        
        while not state.game_over:
            obs = env.get_obs(state)
            mask = env.get_action_mask(state)
            action_idx = int(predict_jit(params, obs, mask))
            
            # (Formatting logic omitted for brevity, essentially same as original)
            if action_idx < env.NUM_KEEP_ACTIONS:
                print(f"👉 KEEP: {format_keep_pattern(env.KEEP_PATTERNS[action_idx])}")
            elif action_idx < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS:
                idx = action_idx - env.NUM_KEEP_ACTIONS
                r, c = idx // 4, idx % 4
                print(f"🎯 SCORE: {ROWS[r]} in {COLS[c]}")
            else:
                idx = action_idx - env.NUM_KEEP_ACTIONS - env.NUM_SCORE_ACTIONS
                print(f"📢 ANNOUNCE: {ROWS[idx]}")
                
            k_step, rng = jax.random.split(rng)
            state, _, _, _, _ = step_jit(k_step, state, action_idx)
            
        render_board(state.board)
        print(f"🏁 FINAL SCORE: {env.calculate_total_score(state.board)}")

    return f.getvalue()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
        
    print(f"🚀 Loading model: {MODEL_PATH}")
    network, params = load_model(MODEL_PATH)
    
    # ── Auto-Detect Input Dimension & Architecture ──
    try:
        # 1. Detect Input Dimension (from first layer kernel)
        # Search for 'kernel' in the first layer (usually Dense_0)
        def find_input_dim(p):
            # Try to find the kernel of the very first layer
            # Hierarchy: params -> Dense_0 -> kernel
            if 'Dense_0' in p and 'kernel' in p['Dense_0']:
                return p['Dense_0']['kernel'].shape[0]
            # Recursion if nested
            for k in p.keys():
                if isinstance(p[k], (dict, FrozenDict, object)) and hasattr(p[k], 'keys'):
                    res = find_input_dim(p[k])
                    if res: return res
            return None

        # Helper to traverse FrozenDict if needed
        def to_dict(d):
            if hasattr(d, 'unfreeze'): return d.unfreeze()
            return d
            
        params_dict = to_dict(params)
        # In restored structure, it's usually params['params']['Dense_0']...
        # But depending on how it was saved/loaded, it might vary.
        # Let's search generally.
        
        # We need to find the specific "actor" and "critic" branches if they exist, 
        # or infer from flat Dense_0, Dense_1... sequence.
        # Our ActorCritic is defined as:
        # actor = x -> Dense_0 -> Dense_1 -> ... -> logits
        # critic = x -> Dense_X -> ... -> value
        
        # Actually, Flax names layers sequentially by default: Dense_0, Dense_1...
        # If actor_layers=[512, 512, 256] (3 layers) -> Dense_0, Dense_1, Dense_2
        # Then actor output logits -> Dense_3
        # Then critic starts -> Dense_4...
        
        # It's tricky to distinguish the boundary without metadata.
        # Retaining hardcoded layers for now but fixing the Obs Dim.
        
        # Recalculate input dim specific to params structure
        # We just need any Kernel that connects to Input-X.
        # `Dense_0` is the first layer of Actor.
        # `Dense_4` (if 3 actor layers + 1 logit) is first layer of Critic.
        # Both take `x` (observation) as input.
        
        # Let's verify Dense_0 exists
        if 'params' in params_dict and 'Dense_0' in params_dict['params']:
            trained_obs_dim = params_dict['params']['Dense_0']['kernel'].shape[0]
        else:
            # Try to find ANY kernel
            trained_obs_dim = find_input_dim(params_dict)
            
        if trained_obs_dim is None:
             raise ValueError("Could not detect input dimension from parameters.")

        print(f"🧠 Detected Model Input Dimension: {trained_obs_dim}")
        print(f"🌍 Current Environment Obs Dimension: {env.OBS_SIZE}")
        
        if trained_obs_dim > env.OBS_SIZE:
            print("❌ Model expects MORE features than environment provides. Cannot run.")
            return
        elif trained_obs_dim < env.OBS_SIZE:
            print(f"⚠️  Model expects fewer features. Observations will be sliced to [: {trained_obs_dim}].")
        else:
            print("✅ Dimensions match perfectly.")
            
    except Exception as e:
        print(f"❌ Failed to inspect model parameters: {e}")
        # Default fallback if inspection fails (unlikely)
        trained_obs_dim = env.OBS_SIZE

    print(f"🔥 Starting {TOTAL_GAMES:,} game evaluation...")
    
    all_scores = []
    all_boards = []
    all_fills = []
    
    best_score = -1
    best_seed = 0
    master_key = jax.random.PRNGKey(42)
    
    t_start = time.time()
    
    # Pre-compile the runner with the specific obs_dim
    # We pass obs_dim as static argument to run_batch -> run_single_game
    
    for b in range(TOTAL_GAMES // BATCH_SIZE):
        k_batch, master_key = jax.random.split(master_key)
        batch_integers = np.random.randint(0, 2**30, size=BATCH_SIZE) + (b * BATCH_SIZE)
        batch_keys = jax.vmap(jax.random.PRNGKey)(jnp.array(batch_integers))
        
        # Run batch
        boards, scores, fill_turns = run_batch(batch_keys, params, trained_obs_dim)
        
        # Move to CPU
        scores_np = np.array(scores)
        boards_np = np.array(boards)
        fills_np = np.array(fill_turns)
        
        all_scores.append(scores_np)
        all_boards.append(boards_np)
        all_fills.append(fills_np)
        
        b_max_idx = np.argmax(scores_np)
        b_max = scores_np[b_max_idx]
        
        if b_max > best_score:
            best_score = b_max
            best_seed = batch_integers[b_max_idx]
            print(f"   [{b+1}] New Record: {best_score} (Seed: {best_seed})")
        elif (b+1) % 5 == 0:
            print(f"   [{b+1}] Max: {b_max}")
                
    elapsed = time.time() - t_start
    print(f"✅ Finished in {elapsed:.2f}s ({TOTAL_GAMES/elapsed:.0f} games/s)")
    
    stats = {
        'scores': np.concatenate(all_scores),
        'boards': np.concatenate(all_boards),
        'fill_turns': np.concatenate(all_fills),
        'model_name': os.path.basename(MODEL_PATH),
        'best_seed': int(best_seed),
        'obs_dim': trained_obs_dim
    }
    
    print("🎥 Capturing best game log...")
    log = capture_game_log(MODEL_PATH, int(best_seed), trained_obs_dim)
    
    # Generate filename based on model name
    report_file = os.path.basename(MODEL_PATH).replace(".npz", "_100k_report.md")
    
    print("📝 Writing report...")
    report = generate_report(stats, log)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"✨ Done! Saved to `{report_file}`")

if __name__ == "__main__":
    main()
