"""
Evaluate JAX Jamb Agent - 100k Games (V2)
=========================================
Runs 100,000 games on GPU to gather comprehensive statistics.
Specific focus on V2 model.
"""

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
    # Convert C:\Path\To\File.py to /mnt/c/Path/To/File.py
    # Assumes standard C: mount
    wsl_path = "/mnt/c" + script_path[2:].replace("\\", "/")
    
    # Check if we need to escape spaces (WSL acting weird with quotes sometimes)
    # Safest is to quote the whole path
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
from flax.linen.initializers import constant, orthogonal
import jamb_jax as env
from watch_agent_jax import ActorCritic, load_model, ROWS, COLS

# ─── Configuration ───────────────────────────────────────────────────
TOTAL_GAMES = 100_000
BATCH_SIZE = 5_000   # Slightly smaller to ensure VRAM safety with tracking
MODEL_DIR = "models/jamb_jax_v2_crazyandfast"
REPORT_FILENAME = "jamb_crazyandfast_100k_report.md"
REPORT_TITLE = "Jamb Agent (Crazy and Fast) Evaluation Report"

# ─── Evaluation Logic ────────────────────────────────────────────────

@jax.jit
def run_single_game(key, model_params):
    """Run a single game to completion and return stats."""
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
        obs = env.get_obs(state)
        mask = env.get_action_mask(state)
        logits, _ = network.apply(model_params, obs)
        masked_logits = jnp.where(mask, logits, -1e9)
        action = jnp.argmax(masked_logits)
        
        # Step
        k1, k2 = jax.random.split(carry[2]) # extract key
        next_state, _, _, done, _ = env.step(k1, state, action)
        
        # Track column fills
        # Check which cols are full in next_state
        # board is (13, 4). Column filled if no -1s
        cols_filled = jnp.all(next_state.board >= 0, axis=0) # (4,) bool
        
        # Update fill_turns: if currently placeholder (100) AND now filled, set to turn_number
        # Note: turn_number in state is 1-based. 
        # When column fills, it's at the END of likely a Score action.
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
    
    # fill_turns initialized to 100 (placeholder for "not filled")
    # There are 4 columns.
    initial_fills = jnp.full((4,), 100, dtype=jnp.int32)
    
    final_state, _, _, final_fills, _ = jax.lax.while_loop(
        cond_fn, 
        body_fn, 
        (state, 0.0, k_loop, initial_fills, False)
    )
    
    final_score = env.calculate_total_score(final_state.board)
    
    return final_state.board, final_score, final_fills

@jax.jit
def run_batch(keys, model_params):
    """Run a batch of games using provided keys."""
    return jax.vmap(run_single_game, in_axes=(0, None))(keys, model_params)

# ─── Main Analysis ───────────────────────────────────────────────────

def get_latest_model():
    """Find latest model in V2."""
    if os.path.exists(MODEL_DIR):
        files = [f for f in os.listdir(MODEL_DIR) 
                 if f.endswith(".npz") and not f.endswith("_tree.json")]
        if files:
            def get_step(fname):
                try: return int(fname.replace("ckpt_", "").replace(".npz", "").split("_")[-1])
                except: return -1
            files.sort(key=get_step, reverse=True)
            return os.path.join(MODEL_DIR, files[0])
    return None

def generate_report(stats, best_game_log):
    scores = stats['scores']
    boards = stats['boards'] 
    fill_turns = stats['fill_turns'] # (N, 4)
    
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    
    # Percentiles
    p1 = np.percentile(scores, 1)
    p10 = np.percentile(scores, 10)
    p25 = np.percentile(scores, 25)
    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)
    p90 = np.percentile(scores, 90)
    p99 = np.percentile(scores, 99)

    avg_cell_values = np.mean(boards, axis=0) # (13, 4)
    
    # Fill speeds
    # Filter out 100s (though all should be filled if game checks game_over correctly)
    # The game ends when ALL cells are filled (game_over condition).
    # So valid games must have all fill_turns <= 52 roughly.
    # Just averaged them.
    avg_fills = np.mean(fill_turns, axis=0)
    
    report = f"""# {REPORT_TITLE}
**Games:** {TOTAL_GAMES:,} | **Model:** `{stats['model_name']}`  
**Device:** GPU (via WSL2 JAX)

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **{avg_score:.2f}** |
| **Max** | **{max_score:.0f}** |
| Median | {p50:.1f} |
| StdDev | {std_score:.2f} |
| Min | {min_score:.0f} |

### Percentiles
| % | Score |
|---|---|
| 1% | {p1:.0f} |
| 10% | {p10:.0f} |
| 25% | {p25:.0f} |
| 50% | {p50:.0f} |
| 75% | {p75:.0f} |
| 90% | {p90:.0f} |
| 99% | {p99:.0f} |

## ⏱️ Column Completion Speed
Average turn number when the column was fully filled (Lower is faster, but usually constrained by rules).
For 'Up' column, it fills bottom-to-top, so 'faster' means finishing 1s earlier.
Wait, game ends around turn 50-60.

| Column | Avg Turn Filled |
|:---|:---|
| **Down** | {avg_fills[0]:.1f} |
| **Free** | {avg_fills[1]:.1f} |
| **Up** | {avg_fills[2]:.1f} |
| **Anno** | {avg_fills[3]:.1f} |

## 🎲 Average Board Values
(Averaged across 100k games)

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

def capture_game_log(model_path, seed):
    """Replays the game with the given seed and returns the log string."""
    network, params = load_model(model_path)
    
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    
    with redirect_stdout(f):
        print(f"--- Replaying Game with Seed {seed} ---")
        
        # Init
        pk = jax.random.PRNGKey(seed)
        k_reset, k_game = jax.random.split(pk)
        state, obs = env.reset(k_reset)
        
        # Helpers
        from watch_agent_jax import render_board, format_histogram, format_keep_pattern
        
        step_jit = jax.jit(env.step)
        predict_jit = jax.jit(lambda p, o, m: jnp.argmax(jnp.where(m, network.apply(p, o)[0], -1e9)))
        
        rng = k_game
        turn_counter = 1
        
        while not state.game_over:
            print(f"\n⚡ TURN {state.turn_number} (Rolls: {state.rolls_left})")
            print(f"🎲 Dice: {format_histogram(state.dice_hist)}")
            
            mask = env.get_action_mask(state)
            action_idx = int(predict_jit(params, obs, mask))
            
            # Print Action
            if action_idx < env.NUM_KEEP_ACTIONS:
                pat = env.KEEP_PATTERNS[action_idx]
                print(f"👉 KEEP: {format_keep_pattern(pat)}")
            elif action_idx < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS:
                idx = action_idx - env.NUM_KEEP_ACTIONS
                r, c = idx // 4, idx % 4
                print(f"� SCORE: {ROWS[r]} in {COLS[c]}")
            else:
                idx = action_idx - env.NUM_KEEP_ACTIONS - env.NUM_SCORE_ACTIONS
                print(f"� ANNOUNCE: {ROWS[idx]}")
                
            k_step, rng = jax.random.split(rng)
            state, obs, _, _, _ = step_jit(k_step, state, action_idx)
            
            if "SCORE" in f.getvalue().splitlines()[-1]: # If last action was score (heuristic check)
                 # Actually just check action index
                 if (action_idx >= env.NUM_KEEP_ACTIONS and 
                     action_idx < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS):
                     print(f"   Current Score: {env.calculate_total_score(state.board)}")
        
        render_board(state.board)
        print(f"🏁 FINAL SCORE: {env.calculate_total_score(state.board)}")

    return f.getvalue()

def main():
    model_path = get_latest_model()
    if not model_path:
        print("❌ No V2 model found!")
        return
        
    print(f"🚀 Loading V2 model: {model_path}")
    network, params = load_model(model_path)
    
    print(f"🔥 Starting 100k game evaluation...")
    print(f"   Batches: {TOTAL_GAMES // BATCH_SIZE} x {BATCH_SIZE}")
    
    all_scores = []
    all_boards = []
    all_fills = []
    
    best_score = -1
    best_seed = 0
    
    # Master seed key
    master_key = jax.random.PRNGKey(42) # Fixed seed for reproducibility of the *batch run*
    
    t_start = time.time()
    
    for b in range(TOTAL_GAMES // BATCH_SIZE):
        k_batch, master_key = jax.random.split(master_key)
        
        # Create seeds for this batch
        # We use simple integers for seeds to be easily logged/reused causes JAX PRNGKey(int) works well
        # We'll generate an array of ints
        seed_subkey = jax.random.fold_in(k_batch, b) # mix batch index
        # Generate random integers using numpy for easy seed tracking
        # Actually, let's just use jax keys
        # But we need "Seed ID" to replay.
        # Let's generate ints.
        batch_integers = np.random.randint(0, 2**30, size=BATCH_SIZE) + (b * BATCH_SIZE) 
        # offset to ensure uniqueness if using deterministic pseudorandom
        
        batch_keys = jax.vmap(jax.random.PRNGKey)(jnp.array(batch_integers))
        
        boards, scores, fill_turns = run_batch(batch_keys, params)
        
        # Move to CPU
        scores_np = np.array(scores)
        boards_np = np.array(boards)
        fills_np = np.array(fill_turns)
        
        all_scores.append(scores_np)
        all_boards.append(boards_np)
        all_fills.append(fills_np)
        
        # Stats
        b_max_idx = np.argmax(scores_np)
        b_max = scores_np[b_max_idx]
        
        if b_max > best_score:
            best_score = b_max
            best_seed = batch_integers[b_max_idx]
            print(f"   [{b+1}] New Record: {best_score} (Seed: {best_seed})")
        else:
            if (b+1) % 5 == 0:
                print(f"   [{b+1}] Max: {b_max}")
                
    elapsed = time.time() - t_start
    print(f"✅ Finished in {elapsed:.2f}s ({TOTAL_GAMES/elapsed:.0f} games/s)")
    
    stats = {
        'scores': np.concatenate(all_scores),
        'boards': np.concatenate(all_boards),
        'fill_turns': np.concatenate(all_fills),
        'model_name': os.path.basename(model_path),
        'best_seed': int(best_seed)
    }
    
    print("🎥 Capturing best game log...")
    log = capture_game_log(model_path, int(best_seed))
    
    print("📝 Writing report...")
    report = generate_report(stats, log)
    
    with open(REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"✨ Done! Saved to `{REPORT_FILENAME}`")

if __name__ == "__main__":
    main()
