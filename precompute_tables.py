"""
Precompute EXACT perfect-play expected scores for Jamb using Dynamic Programming.
Outputs a (924, 6, 13) lookup table used by the environment's actions.

Two steps:
1. Value Iteration on the 462 full-dice states (0..5 rolls).
2. For each of the 924 keep-pattern actions, compute expected value by summing over roll outcomes.

Outputs:
  - keep_patterns.npy: shape (924, 6) — all valid partial keep histograms
  - expected_scores.npy: shape (924, 6, 13) — optimal E[score] for action `keep_pattern` with `rolls_left`
"""

import numpy as np
import time
from math import factorial
from itertools import combinations_with_replacement

NUM_ROWS = 13
MAX_ROLLS = 5

# ─── 1. Enumeration ─────────────────────────────────────────────────

def enumerate_full_states():
    """Returns (462, 6) array of all dice histograms with sum=6."""
    states = []
    for c in combinations_with_replacement(range(6), 6):
        hist = np.zeros(6, dtype=np.int8)
        for val in c:
            hist[val] += 1
        states.append(hist)
    states = sorted(states, key=lambda x: tuple(x))
    return np.array(states, dtype=np.int8)

def enumerate_keep_patterns():
    """Returns (924, 6) array of all dice histograms with sum<=6."""
    patterns = []
    # Iterate length 0 to 6
    for length in range(7):
        for c in combinations_with_replacement(range(6), length):
            hist = np.zeros(6, dtype=np.int8)
            for val in c:
                hist[val] += 1
            patterns.append(hist)
    patterns = sorted(patterns, key=lambda x: tuple(x))
    return np.array(patterns, dtype=np.int8)

# ─── 2. Transition Probability ──────────────────────────────────────

def multinomial_prob(counts):
    # Enforce native python int to avoid numpy int8 overflow issues
    n = int(sum(counts))
    if n == 0: return 1.0
    
    p = factorial(n)
    for c in counts:
        p //= factorial(int(c))
    
    return float(p) / float(6**n)

def get_roll_outcomes_and_probs(n_dice):
    """List of (outcome_hist, prob) for rolling n_dice."""
    outcomes = []
    for c in combinations_with_replacement(range(6), n_dice):
        hist = np.zeros(6, dtype=np.int8)
        for val in c:
            hist[val] += 1
        prob = multinomial_prob(hist)
        outcomes.append((hist, prob))
    return outcomes

# ─── 3. Scoring (Vectorized) ────────────────────────────────────────

def calc_scores_all_states(states):
    """Compute immediate scores for full states. (N, 13)."""
    n_states = len(states)
    scores = np.zeros((n_states, NUM_ROWS), dtype=np.float32)
    
    for i in range(n_states):
        hist = states[i]
        dice = []
        for v, count in enumerate(hist):
            dice.extend([v+1] * count)
        dice = np.array(dice)
        counts = np.zeros(7, dtype=int)
        counts[1:] = hist

        # 1s..6s
        for cat in range(6):
            val = cat + 1
            scores[i, cat] = min(counts[val], 5) * val
        # Max/Min
        scores[i, 6] = np.sum(dice[-5:])
        scores[i, 7] = np.sum(dice[:5])
        # Trips
        for v in range(6, 0, -1):
            if counts[v] >= 3:
                scores[i, 8] = 3*v + 20
                break
        # Straight
        unique = set(dice)
        if {2,3,4,5,6} <= unique: scores[i, 9] = 50
        elif {1,2,3,4,5} <= unique: scores[i, 9] = 45
        # Full/Poker/Yamb
        best = 0
        for v1 in range(6, 0, -1):
            if counts[v1] >= 3:
                for v2 in range(6, 0, -1):
                    if v1==v2: continue
                    if counts[v2] >= 2:
                        best = max(best, 3*v1 + 2*v2 + 40)
        scores[i, 10] = best
        for v in range(6, 0, -1):
            if counts[v] >= 4:
                scores[i, 11] = 4*v + 50
                break
        for v in range(6, 0, -1):
            if counts[v] >= 5:
                scores[i, 12] = 5*v + 60
                break
                
    return scores

# ─── 4. Main ────────────────────────────────────────────────────────

def main():
    start_total = time.time()
    
    # 1. Setup States (Full Hand)
    states = enumerate_full_states()
    N_states = len(states)
    print(f"Full states (sum=6): {N_states}")
    
    state_to_idx = {tuple(s): i for i, s in enumerate(states)}
    
    # 2. Setup Keep Patterns (Partial Hands)
    patterns = enumerate_keep_patterns()
    N_patterns = len(patterns)
    print(f"Keep patterns (sum<=6): {N_patterns}")
    np.save("keep_patterns.npy", patterns)
    
    # 3. Precompute Transitions
    print("Precalculating transitions...")
    transitions = [get_roll_outcomes_and_probs(n) for n in range(7)]
    
    # 4. Value Iteration on STATES
    # V[state_idx, rolls_left, category]
    V = np.zeros((N_states, 6, NUM_ROWS), dtype=np.float32)
    
    # Base case: Rolls=0 -> value is immediate score
    V[:, 0, :] = calc_scores_all_states(states)
    print(f"Base Max Score (r=0): {np.max(V[:, 0, :])}")

    # Check Transitions
    for i in range(7):
        p_sum = sum(p for (_, p) in transitions[i])
        print(f"  Transitions[roll={i}]: sum(prob) = {p_sum:.4f}")

    print("Running Value Iteration on states (1..5 rolls)...")
    import itertools
    
    for r in range(1, 6):
        # V[s, r] = max_{keep <= s} E[ V[keep+outcome, r-1] ]
        # Optimization: We iterate states. For each state, we iterate valid keeps.
        
        for idx in range(N_states):
            hist = states[idx]
            
            # Valid keeps are sub-multisets
            ranges = [range(c + 1) for c in hist]
            
            # Start with "Keep All" (value = V[idx, r-1])
            best_values = V[idx, r-1, :].copy()
            
            for keep_tuple in itertools.product(*ranges):
                keep = np.array(keep_tuple, dtype=np.int8)
                n_kept = np.sum(keep)
                n_roll = 6 - n_kept
                if n_roll == 0: continue
                
                # Expected value of this keep action
                acc = np.zeros(NUM_ROWS, dtype=np.float32)
                for (out_hist, prob) in transitions[n_roll]:
                    res_hist = keep + out_hist
                    res_idx = state_to_idx[tuple(res_hist)]
                    acc += prob * V[res_idx, r-1, :]
                    
                best_values = np.maximum(best_values, acc)
            
            V[idx, r, :] = best_values

    print(f"  State Values computed. (Elapsed: {time.time()-start_total:.1f}s)")
    
    # 5. Populate Result Table for ACTIONS
    # result[pattern_idx, rolls_left, category]
    # For a pattern P with R rolls left:
    # Value = E[ V(P + outcome, R-1) ]  <-- wait, R rolls left *before* this roll?
    # In JambEnv: `rl` is rolls remaining *before* taking action.
    # If I choose keep pattern P, I commit to rolling (6-sum(P)) dice.
    # After that roll, I will have `rl-1` rolls left.
    # So Value = E[ V(resulting_state, rl-1) ]
    
    final_table = np.zeros((N_patterns, 6, NUM_ROWS), dtype=np.float32)
    
    print("Populating Keep Pattern table...")
    
    for p_idx in range(N_patterns):
        pattern = patterns[p_idx]
        n_kept = np.sum(pattern)
        n_roll = 6 - n_kept
        
        # If pattern is full (sum=6), rolling 0 dice means we keep state.
        # But `rolls_left` logic:
        # If I have 1 roll left, and keeping full state, I move to state with 0 rolls left.
        # Value = V[state_idx, rl-1] (if n_roll=0).
        
        # Pre-calc transitions for this pattern
        outcomes = transitions[n_roll]
        
        for rl in range(1, 6):
            acc = np.zeros(NUM_ROWS, dtype=np.float32)
            for (out_hist, prob) in outcomes:
                res_hist = pattern + out_hist
                res_idx = state_to_idx[tuple(res_hist)]
                acc += prob * V[res_idx, rl-1, :]
            
            final_table[p_idx, rl, :] = acc

    # Handle rl=0 case: technically undefined for partial patterns (must roll).
    # But if pattern is full, value is score.
    # JambEnv masks partial patterns if rolls=0 (logic: rolls_left<=0 -> no roll).
    # We'll fill with 0 or immediate score if full.
    for p_idx in range(N_patterns):
        pattern = patterns[p_idx]
        if np.sum(pattern) == 6:
            # It's a full state
            s_idx = state_to_idx[tuple(pattern)]
            final_table[p_idx, 0, :] = V[s_idx, 0, :]
            
    np.save("expected_scores.npy", final_table)
    print(f"  Saved expected_scores.npy ({final_table.nbytes/1024:.1f} KB)")
    print(f"Total time: {time.time()-start_total:.1f}s")
    
    # Sanity Check
    # Keep Nothing (0,0,0,0,0,0), 3 rolls left.
    # Should equate to E[ V(random_hand, 2) ]
    # i.e. "Start Game Value"
    try:
        pid = 0 # (0,0,0,0,0,0) is first
        print("\nSanity Check (Keep Nothing, 3 rolls left):")
        print(f"  Yamb: {final_table[pid, 3, 12]:.1f}")
        print(f"  Max:  {final_table[pid, 3, 6]:.1f}")
    except:
        pass

if __name__ == "__main__":
    main()
