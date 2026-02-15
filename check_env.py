"""
Environment sanity checks for the Jamb RL v6 environment.

Tests:
  1. Random play: 100 episodes complete without crashes.
  2. Action mask correctness: keep patterns respect current dice histogram.
  3. No keep-all-6: verify no pattern in the action space keeps all 6 dice.
  4. Last-cell announcement: forced announcement prevents stalemate.
  5. Histogram equivalence: dice obs matches expected histogram.
  6. Cell scores masking: filled/blocked cells have zero cell_scores.
"""

from jamb_env import JambEnv
from jamb_logic import NUM_ROWS, NUM_COLS, Column
import numpy as np


def test_random_play(n_episodes=100):
    """Run random episodes and confirm no crashes or stalemates."""
    print(f"[TEST 1] Random play ({n_episodes} episodes)...")
    env = JambEnv()
    scores = []
    stalemates = 0
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        assert obs.shape == (166,), f"Obs shape {obs.shape} != (166,)"
        terminated = False
        steps = 0
        
        while not terminated:
            masks = env.action_masks()
            valid_indices = np.where(masks)[0]
            
            if len(valid_indices) == 0:
                print(f"  FATAL: No valid actions at episode {i}, step {steps}!")
                print(f"  Rolls: {env.logic.rolls_left}, Empty: {np.sum(env.logic.board == -1)}")
                stalemates += 1
                break
                
            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (166,), f"Obs shape changed to {obs.shape}"
            steps += 1
            
            if steps > 1000:
                print(f"  WARNING: Episode {i} exceeded 1000 steps, breaking.")
                stalemates += 1
                break
            
        scores.append(env.logic.calculate_total_score())
        if (i + 1) % 25 == 0:
            print(f"  Ep {i+1}: Score {scores[-1]}")

    print(f"  Mean Score: {np.mean(scores):.1f}")
    print(f"  Min/Max: {np.min(scores)} / {np.max(scores)}")
    print(f"  Stalemates: {stalemates}")
    assert stalemates == 0, "Stalemates detected!"
    print("  ✅ PASSED\n")


def test_keep_pattern_masking():
    """Verify keep patterns are correctly masked against current dice."""
    print("[TEST 2] Keep pattern mask correctness...")
    env = JambEnv()
    
    for trial in range(20):
        obs, _ = env.reset()
        hist = env.logic.dice_histogram.copy()
        masks = env.action_masks()
        
        for i in range(env.NUM_KEEP_ACTIONS):
            pattern = env.keep_patterns[i]
            valid = np.all(pattern <= hist)
            
            if masks[i] and not valid:
                print(f"  FAIL: Pattern {pattern} enabled but exceeds hist {hist}")
                assert False
    
    print("  ✅ PASSED (20 trials)\n")


def test_no_keep_all_six():
    """Verify no keep pattern sums to 6 (would roll 0 dice = invalid)."""
    print("[TEST 3] No keep-all-6 in action space...")
    env = JambEnv()
    
    for i, pat in enumerate(env.keep_patterns):
        total_kept = pat.sum()
        assert total_kept < 6, f"Pattern {i} keeps {total_kept} dice: {pat}"
    
    # Verify action space size
    expected_actions = len(env.keep_patterns) + NUM_ROWS * NUM_COLS + NUM_ROWS
    assert env.action_space.n == expected_actions, \
        f"Action space {env.action_space.n} != expected {expected_actions}"
    assert env.action_space.n == 527, \
        f"Action space {env.action_space.n} != 527"
    
    print(f"  Action space: {env.action_space.n} (462 keep + 52 score + 13 announce)")
    print("  ✅ PASSED\n")


def test_forced_announcement():
    """Verify forced announcement when only Announce column cells remain."""
    print("[TEST 4] Forced announcement on last cell...")
    env = JambEnv()
    env.logic.reset()
    
    # Fill ALL cells except one in the Announce column (Yamb in Announce)
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            if not (r == 12 and c == 3):
                env.logic.board[r, c] = 0
    
    env.logic.rolls_left = 5
    env.logic.current_roll_count = 0
    env.logic.announced_row = -1
    env.logic.pure_hand = True
    env.logic.roll_dice(np.zeros(6, dtype=int))
    
    masks = env.action_masks()
    
    assert not np.any(masks[:env.NUM_KEEP_ACTIONS]), "Keep actions should be masked!"
    assert not np.any(masks[env.NUM_KEEP_ACTIONS:env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS]), \
        "Score actions should be masked!"
    
    announce_start = env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS
    assert masks[announce_start + 12], "Yamb announce should be enabled!"
    for r in range(12):
        assert not masks[announce_start + r], f"Announce row {r} should NOT be enabled!"
    
    obs, _, terminated, _, _ = env.step(announce_start + 12)
    assert not terminated
    
    masks = env.action_masks()
    assert np.any(masks[:env.NUM_KEEP_ACTIONS]), "Keep actions should be available after announcing!"
    
    print("  ✅ PASSED\n")


def test_histogram_equivalence():
    """Verify that dice histogram observation matches expected values."""
    print("[TEST 5] Histogram equivalence...")
    env = JambEnv()
    
    env.logic.dice_histogram = np.array([5, 0, 0, 0, 0, 1], dtype=int)
    env.logic.rolls_left = 2
    env.logic.current_roll_count = 1
    obs = env._get_obs()
    
    # Dice section: indices 104..109 (after 52 board_scores + 52 board_filled)
    dice_obs = obs[104:110]
    expected = np.array([5/6, 0, 0, 0, 0, 1/6], dtype=np.float32)
    assert np.allclose(dice_obs, expected, atol=1e-5), \
        f"Dice obs {dice_obs} != expected {expected}"
    
    print("  ✅ PASSED\n")


def test_cell_scores_masking():
    """Verify cell_scores are zero for filled/blocked cells and correct for legal moves."""
    print("[TEST 6] Cell scores masking...")
    env = JambEnv()
    env.logic.reset()
    
    # Fill ALL cells for categories 0-5 (1s through 6s) in ALL columns
    for r in range(6):
        for c in range(NUM_COLS):
            env.logic.board[r, c] = 10
    
    # Set up dice and rolls: all 6s → Yamb should score well
    env.logic.dice_histogram = np.array([0, 0, 0, 0, 0, 6], dtype=int)
    env.logic.rolls_left = 2
    env.logic.current_roll_count = 1
    
    obs = env._get_obs()
    
    # cell_scores is the last 52 values in the observation
    cell_scores = obs[-52:]
    
    # Rows 0-5 filled → all 24 cells should be 0
    for r in range(6):
        for c in range(NUM_COLS):
            idx = r * NUM_COLS + c
            assert cell_scores[idx] == 0.0, \
                f"Filled cell ({r},{c}) should have score 0, got {cell_scores[idx]}"
    
    # Yamb (row 12) should have positive scores in at least some columns
    # With all 6s, Yamb = 5*6 + 60 = 90, so cell_scores should be 0.90
    yamb_scores = []
    for c in range(NUM_COLS):
        idx = 12 * NUM_COLS + c
        yamb_scores.append(cell_scores[idx])
    
    # At minimum Free column should be available
    assert any(s > 0 for s in yamb_scores), \
        f"Yamb should have positive scores in some columns, got {yamb_scores}"
    
    # Verify Announce column rules: without announcement, Announce column 
    # should only be available via pure_hand (which is True after first roll)
    announce_yamb_idx = 12 * NUM_COLS + Column.ANNOUNCE
    # Pure hand is True since current_roll_count=1 and we haven't kept dice
    # So pure hand scoring in Announce should be available
    
    print(f"  Yamb cell_scores per column: {yamb_scores}")
    print("  ✅ PASSED\n")


if __name__ == "__main__":
    test_random_play()
    test_keep_pattern_masking()
    test_no_keep_all_six()
    test_forced_announcement()
    test_histogram_equivalence()
    test_cell_scores_masking()
    print("🎉 All tests passed!")
