import argparse
import time
import numpy as np
import os
from sb3_contrib import MaskablePPO
from jamb_env import JambEnv


def render_board(logic):
    rows = ["1s", "2s", "3s", "4s", "5s", "6s", "Max", "Min", "T", "K", "F", "P", "Y"]
    cols = ["Down", "Free", "Up", "Anno"]
    
    header = "      " + " ".join([f"{c:^8}" for c in cols])
    print("\n" + header)
    print("      " + "-" * 36)
    
    for r_idx, row_name in enumerate(rows):
        line = f"{row_name:>4} | "
        for c_idx in range(4):
            val = logic.board[r_idx, c_idx]
            if val == -1:
                display = "."
            else:
                display = str(val)
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


def format_keep_pattern(pattern, current_hist):
    """Format a keep pattern as a list of dice."""
    kept_dice = []
    for val_idx in range(6):
        count = int(pattern[val_idx])
        kept_dice.extend([val_idx + 1] * count)
    return str(kept_dice) if kept_dice else "nothing"


def watch_agent(model_path, delay=0.5):
    env = JambEnv()
    
    if not os.path.exists(model_path) and not model_path.endswith(".zip"):
        model_path += ".zip"
        
    try:
        model = MaskablePPO.load(model_path)
        print(f"\n🚀 LOADED AGENT: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    obs, _ = env.reset()
    terminated = False
    last_turn_printed = -1
    
    ROWS = ["1s", "2s", "3s", "4s", "5s", "6s", "Max", "Min", "T", "K", "F", "P", "Y"]
    COLS = ["Down", "Free", "Up", "Anno"]

    print("\n" + "="*40)
    print("      JAMB RL AGENT - LIVE GAME (v4)")
    print("="*40)
    
    while not terminated:
        turn_now = env.logic.turn_number
        rolls_before = env.logic.rolls_left
        hist_before = env.logic.dice_histogram.copy()

        if turn_now != last_turn_printed:
            print(f"\n--- TURN {turn_now} ---")
            last_turn_printed = turn_now

        print(f"🎲 Dice: [{format_histogram(hist_before)}] (Rolls Remaining: {rolls_before})")
        
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if action < env.NUM_KEEP_ACTIONS:
            pattern = env.keep_patterns[action]
            print(f"👉 ACTION: ROLL (Kept: {format_keep_pattern(pattern, hist_before)})")
        elif action < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS:
            idx = action - env.NUM_KEEP_ACTIONS
            r_idx, c_idx = idx // 4, idx % 4
            print(f"👉 ACTION: SCORE {ROWS[r_idx]} in {COLS[c_idx]}")
            render_board(env.logic)
            print(f"🏆 Total Score: {env.logic.calculate_total_score()}")
            print("-" * 40)
        else:
            row = action - env.NUM_KEEP_ACTIONS - env.NUM_SCORE_ACTIONS
            print(f"👉 ACTION: ANNOUNCE {ROWS[row]}")
            
        if delay > 0:
            time.sleep(delay)

    print(f"\n🏁 GAME OVER!")
    print(f"✨ Final Score: {env.logic.calculate_total_score()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Auto-detect latest model (v5a/b/c > v4 > v3)
    latest_model = "jamb_ppo_v5a_final"
    
    search_order = [
        ("v5", "jamb_ppo_v5"),
        ("v5a", "jamb_ppo_v5a"), ("v5b", "jamb_ppo_v5b"), ("v5c", "jamb_ppo_v5c"),
        ("v4", "jamb_ppo_v4"), ("v3", "jamb_ppo_v3"), ("v2", "jamb_ppo_v2"),
    ]
    for version, prefix in search_order:
        d = f"models/jamb_ppo_{version}"
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".zip")]
            if files:
                def get_step_count(fname):
                    if "final" in fname:
                        return float('inf')
                    try:
                        # Extract the step number from "..._12345_steps.zip"
                        parts = fname.split("_steps")
                        if len(parts) > 1:
                            num_part = parts[0].split("_")[-1]
                            return int(num_part)
                    except ValueError:
                        pass
                    return -1

                files.sort(key=get_step_count)
                latest_model = os.path.join(d, files[-1])
                break

    parser.add_argument("--model", type=str, default=latest_model, help="Path to model file")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps")
    args = parser.parse_args()
    
    watch_agent(args.model, args.delay)
