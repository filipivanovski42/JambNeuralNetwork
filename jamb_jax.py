"""
Jamb RL Environment — Pure JAX Implementation
==============================================
Fully JIT-compatible, vmap-able game environment for GPU-accelerated training.
Replicates the logic of jamb_env.py / jamb_logic.py using only jax.numpy.

Action space (527 discrete):
  0..461   : Keep patterns (roll remaining dice)
  462..513 : Score actions  (row*4 + col)
  514..526 : Announce actions (row 0-12)

Observation space (166 floats):
  board_scores[52] + board_filled[52] + dice_hist[6] + scalars[4] + cell_scores[52]
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial
import os

# ─── Constants ───────────────────────────────────────────────────────
NUM_ROWS = 13
NUM_COLS = 4
NUM_DICE = 6
NUM_SCORE_ACTIONS = NUM_ROWS * NUM_COLS  # 52
NUM_ANNOUNCE_ACTIONS = NUM_ROWS          # 13

# Column indices
COL_DOWN = 0
COL_FREE = 1
COL_UP = 2
COL_ANNOUNCE = 3

# Row indices
ROW_ONES = 0; ROW_TWOS = 1; ROW_THREES = 2; ROW_FOURS = 3
ROW_FIVES = 4; ROW_SIXES = 5; ROW_MAX = 6; ROW_MIN = 7
ROW_TRIPS = 8; ROW_STRAIGHT = 9; ROW_FULL = 10; ROW_POKER = 11; ROW_YAMB = 12


# ─── Game State ──────────────────────────────────────────────────────
class JambState(NamedTuple):
    board: jnp.ndarray          # (13, 4) int32, -1 = empty
    dice_hist: jnp.ndarray      # (6,) int8, counts of each face value
    rolls_left: jnp.int32
    turn_number: jnp.int32
    announced_row: jnp.int32    # -1 = none
    pure_hand: jnp.bool_
    current_roll_count: jnp.int32
    game_over: jnp.bool_


# ─── Preload keep_patterns.npy ───────────────────────────────────────
def _load_keep_patterns():
    """Load keep_patterns from disk and filter out keep-all-6."""
    base = os.path.dirname(os.path.abspath(__file__))
    kp = np.load(os.path.join(base, "keep_patterns.npy"))
    valid = kp.sum(axis=1) < 6
    return jnp.array(kp[valid], dtype=jnp.int8)  # (462, 6)

KEEP_PATTERNS = _load_keep_patterns()
NUM_KEEP_ACTIONS = KEEP_PATTERNS.shape[0]  # 462
TOTAL_ACTIONS = NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS + NUM_ANNOUNCE_ACTIONS  # 527
OBS_SIZE = 178


# ─── Scoring Functions ───────────────────────────────────────────────

def _calc_score_for_row(dice_hist: jnp.ndarray, row: int) -> jnp.int32:
    """Calculate score for a specific row given dice histogram. Pure JAX."""
    counts = dice_hist  # (6,) where counts[i] = number of dice showing (i+1)

    # Expand to sorted dice array for sum operations
    # dice values: [1]*counts[0] + [2]*counts[1] + ... + [6]*counts[5]
    # We need sum of top-5 and bottom-5
    # Build cumulative for top/bottom 5

    # For numbers rows (0-5): min(counts[row], 5) * (row+1)
    def score_numbers(r):
        val = jnp.int32(r + 1)
        return jnp.int32(jnp.minimum(counts[r], 5) * val)

    # For Max: sum of 5 highest dice
    def score_max():
        total_sum = jnp.int32(jnp.sum(jnp.arange(1, 7) * counts))
        vals = jnp.arange(1, 7)
        has_dice = counts > 0
        min_val = jnp.int32(jnp.where(has_dice, vals, 7).min())
        return total_sum - min_val

    # For Min: sum of 5 lowest dice
    def score_min():
        total_sum = jnp.int32(jnp.sum(jnp.arange(1, 7) * counts))
        has_dice = counts > 0
        vals = jnp.arange(1, 7)
        max_val = jnp.int32(jnp.where(has_dice, vals, 0).max())
        return total_sum - max_val

    # For Trips: find highest value with count >= 3
    def score_trips():
        has_trips = counts >= 3
        vals = jnp.arange(1, 7)
        best_val = jnp.int32(jnp.where(has_trips, vals, 0).max())
        return jnp.int32(jnp.where(best_val > 0, 3 * best_val + 20, 0))

    # For Straight
    def score_straight():
        has = counts > 0
        has_small = jnp.all(has[:5])
        has_large = jnp.all(has[1:])
        return jnp.int32(jnp.where(has_large, 50, jnp.where(has_small, 45, 0)))

    # For Full House: 3 of one kind + 2 of another (different values)
    def score_full():
        vals = jnp.arange(1, 7)
        # For each pair (v1, v2) where v1 != v2, check counts[v1-1]>=3 and counts[v2-1]>=2
        # Compute all possible scores and take max
        # Use meshgrid approach
        v1_idx = jnp.arange(6)
        v2_idx = jnp.arange(6)
        # Create all pairs
        v1_grid, v2_grid = jnp.meshgrid(v1_idx, v2_idx, indexing='ij')  # (6,6)
        v1_grid = v1_grid.flatten()  # (36,)
        v2_grid = v2_grid.flatten()
        different = v1_grid != v2_grid
        has_3 = counts[v1_grid] >= 3
        has_2 = counts[v2_grid] >= 2
        valid = different & has_3 & has_2
        scores = jnp.int32((3 * (v1_grid + 1) + 2 * (v2_grid + 1)) + 40)
        scores = jnp.where(valid, scores, jnp.int32(0))
        return jnp.int32(scores.max())

    # For Poker: 4 of a kind
    def score_poker():
        has_poker = counts >= 4
        vals = jnp.arange(1, 7)
        best_val = jnp.int32(jnp.where(has_poker, vals, 0).max())
        return jnp.int32(jnp.where(best_val > 0, 4 * best_val + 50, 0))

    # For Yamb: 5 of a kind
    def score_yamb():
        has_yamb = counts >= 5
        vals = jnp.arange(1, 7)
        best_val = jnp.int32(jnp.where(has_yamb, vals, 0).max())
        return jnp.int32(jnp.where(best_val > 0, 5 * best_val + 60, 0))

    # Use jax.lax.switch for row dispatch
    row = jnp.int32(row)
    score = jax.lax.switch(row, [
        lambda: score_numbers(0),   # 0 - Ones
        lambda: score_numbers(1),   # 1 - Twos
        lambda: score_numbers(2),   # 2 - Threes
        lambda: score_numbers(3),   # 3 - Fours
        lambda: score_numbers(4),   # 4 - Fives
        lambda: score_numbers(5),   # 5 - Sixes
        score_max,                  # 6 - Max
        score_min,                  # 7 - Min
        score_trips,                # 8 - Trips
        score_straight,             # 9 - Straight
        score_full,                 # 10 - Full
        score_poker,                # 11 - Poker
        score_yamb,                 # 12 - Yamb
    ])
    return jnp.int32(score)


def calc_all_scores(dice_hist: jnp.ndarray) -> jnp.ndarray:
    """Calculate scores for ALL 13 rows at once. Returns (13,) int32."""
    return jax.vmap(lambda r: _calc_score_for_row(dice_hist, r))(jnp.arange(13))


# ─── Board Scoring (Final Score) ─────────────────────────────────────

def calculate_total_score(board: jnp.ndarray) -> jnp.int32:
    """Calculate total game score from a filled board. (13,4) -> int."""
    def col_score(c):
        col = board[:, c]

        # Section 1: Numbers (rows 0-5)
        nums = jnp.where(col[:6] >= 0, col[:6], 0)
        sum_1_6 = jnp.sum(nums)
        bonus = jnp.where(sum_1_6 >= 60, 30, 0)
        sec1 = sum_1_6 + bonus

        # Section 2: (Max - Min) * Ones
        val_max = col[ROW_MAX]
        val_min = col[ROW_MIN]
        val_ones = col[ROW_ONES]
        all_filled = (val_max >= 0) & (val_min >= 0) & (val_ones >= 0)
        diff = jnp.maximum(val_max - val_min, 0)
        sec2 = jnp.where(all_filled, diff * val_ones, 0)

        # Section 3: Combos (rows 8-12)
        combos = jnp.where(col[8:13] >= 0, col[8:13], 0)
        sec3 = jnp.sum(combos)

        return sec1 + sec2 + sec3

    return jnp.sum(jax.vmap(col_score)(jnp.arange(4)))


# ─── Available Actions (Scoring Moves) ───────────────────────────────

def get_valid_score_mask(state: JambState) -> jnp.ndarray:
    """Returns (13, 4) bool mask of valid scoring cells."""
    board = state.board
    is_empty = board == -1  # (13, 4)

    # Column DOWN (col 0): row r valid if empty AND (r==0 OR row r-1 filled)
    down_valid = is_empty[:, COL_DOWN]
    prev_filled = jnp.concatenate([jnp.array([True]), ~is_empty[:-1, COL_DOWN]])
    down_valid = down_valid & prev_filled

    # Column FREE (col 1): any empty cell
    free_valid = is_empty[:, COL_FREE]

    # Column UP (col 2): row r valid if empty AND (r==12 OR row r+1 filled)
    up_valid = is_empty[:, COL_UP]
    next_filled = jnp.concatenate([~is_empty[1:, COL_UP], jnp.array([True])])
    up_valid = up_valid & next_filled

    # Column ANNOUNCE (col 3): complex rules
    anno_empty = is_empty[:, COL_ANNOUNCE]
    # If announced: only that row
    has_announcement = state.announced_row >= 0
    announced_mask = jnp.arange(NUM_ROWS) == state.announced_row
    # If pure hand and no announcement: any empty cell
    pure_hand_mask = anno_empty & state.pure_hand & (~has_announcement)
    # If announced: only the announced row (if empty)
    anno_mask = anno_empty & announced_mask & has_announcement
    announce_valid = pure_hand_mask | anno_mask

    mask = jnp.stack([down_valid, free_valid, up_valid, announce_valid], axis=1)

    # If announced, FORCE only the announced cell
    announced_force = jnp.zeros((NUM_ROWS, NUM_COLS), dtype=jnp.bool_)
    announced_force = announced_force.at[state.announced_row, COL_ANNOUNCE].set(True)
    announced_force = announced_force & anno_empty[state.announced_row]
    # Use the forced mask when announced
    # When announced_row >= 0, override entire mask
    mask = jnp.where(has_announcement, announced_force, mask)

    return mask


# ─── Full Action Mask ─────────────────────────────────────────────────

def get_action_mask(state: JambState) -> jnp.ndarray:
    """Returns (527,) bool mask of all valid actions."""
    mask = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_)

    # ── Keep actions (0..461) ──
    # Valid when rolls_left > 0 and pattern <= current histogram
    can_roll = state.rolls_left > 0
    valid_keeps = jnp.all(KEEP_PATTERNS <= state.dice_hist, axis=1)  # (462,)
    keep_mask = valid_keeps & can_roll
    mask = mask.at[:NUM_KEEP_ACTIONS].set(keep_mask)

    # ── Score actions (462..513) ──
    score_mask_2d = get_valid_score_mask(state)  # (13, 4)
    score_mask_flat = score_mask_2d.flatten()  # (52,)
    mask = mask.at[NUM_KEEP_ACTIONS:NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS].set(score_mask_flat)

    # ── Announce actions (514..526) ──
    can_announce = (state.current_roll_count == 1) & (state.announced_row < 0)
    anno_empty = state.board[:, COL_ANNOUNCE] == -1  # (13,)
    announce_mask = anno_empty & can_announce
    mask = mask.at[NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS:].set(announce_mask)

    # ── Force announcement if all remaining cells are in Announce column ──
    empty_cells = state.board == -1  # (13, 4)
    empty_non_announce = empty_cells[:, :3].any()
    all_in_announce = ~empty_non_announce & empty_cells[:, COL_ANNOUNCE].any()
    force_announce = all_in_announce & can_announce
    # If forcing, clear everything except announce actions
    forced_mask = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_)
    forced_mask = forced_mask.at[NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS:].set(announce_mask)
    mask = jnp.where(force_announce, forced_mask, mask)

    # ── Safety fallback ──
    mask = jnp.where(mask.any(), mask, jnp.ones(TOTAL_ACTIONS, dtype=jnp.bool_))

    return mask


# ─── Observation ──────────────────────────────────────────────────────

def get_obs(state: JambState) -> jnp.ndarray:
    """Build 178-dimensional observation vector."""
    board = state.board  # (13, 4) int32, -1 = empty
    board_flat = board.flatten().astype(jnp.float32)  # (52,)

    # Board scores (normalised)
    board_scores = jnp.where(board_flat >= 0, board_flat, 0.0) / 100.0

    # Board filled (binary)
    board_filled = (board_flat >= 0).astype(jnp.float32)

    # Dice histogram
    dice_hist = state.dice_hist.astype(jnp.float32) / 6.0

    # Scalars
    rolls_norm = state.rolls_left / 5.0
    anno_norm = jnp.where(state.announced_row >= 0,
                          (state.announced_row + 1) / 13.0, 0.0)
    pure_hand = state.pure_hand.astype(jnp.float32)
    turn_norm = state.turn_number / 52.0

    # Per-cell scores for legal moves
    all_scores = calc_all_scores(state.dice_hist)  # (13,)
    score_mask_2d = get_valid_score_mask(state)     # (13, 4)
    cell_scores_2d = jnp.where(score_mask_2d, all_scores[:, None], 0.0)
    cell_scores = cell_scores_2d.flatten().astype(jnp.float32) / 100.0  # (52,)

    # ── Column section totals (12 features) ──
    # Replace -1 (empty) with 0 for summation
    filled = jnp.where(board >= 0, board, 0).astype(jnp.float32)

    # Numbers sum per column (rows 0-5), normalised by 30 (~max per section)
    nums_sums = jnp.sum(filled[:6, :], axis=0) / 30.0  # (4,)

    # Max-Min spread per column, normalised by 25 (~max spread)
    val_max = filled[ROW_MAX, :]  # (4,)
    val_min = filled[ROW_MIN, :]  # (4,)
    max_filled = board[ROW_MAX, :] >= 0
    min_filled = board[ROW_MIN, :] >= 0
    spread = jnp.where(max_filled & min_filled,
                       jnp.maximum(val_max - val_min, 0.0), 0.0) / 25.0  # (4,)

    # Combos sum per column (rows 8-12), normalised by 250 (~max combos)
    combo_sums = jnp.sum(filled[8:13, :], axis=0) / 250.0  # (4,)

    obs = jnp.concatenate([
        board_scores,                    # 52
        board_filled,                    # 52
        dice_hist,                       # 6
        jnp.array([rolls_norm]),         # 1
        jnp.array([anno_norm]),          # 1
        jnp.array([pure_hand]),          # 1
        jnp.array([turn_norm]),          # 1
        cell_scores,                     # 52
        nums_sums,                       # 4
        spread,                          # 4
        combo_sums,                      # 4
    ])
    return obs  # (178,)


# ─── Roll Dice ────────────────────────────────────────────────────────

def roll_dice(key: jnp.ndarray, state: JambState, keep_hist: jnp.ndarray) -> JambState:
    """Roll dice, keeping the specified histogram. Returns new state."""
    n_kept = jnp.sum(keep_hist)

    # Pure hand logic
    new_pure = jnp.where(
        state.current_roll_count > 0,
        n_kept == 0,  # Pure if keeping nothing
        True           # First roll always pure
    )

    # Roll remaining dice
    n_to_roll = 6 - n_kept
    # Generate n_to_roll random dice (always generate 6 and mask)
    rolls = jax.random.randint(key, shape=(6,), minval=0, maxval=6)
    # Convert to histogram
    new_hist = keep_hist.copy()
    # Add each roll to histogram (only first n_to_roll matter)
    roll_mask = jnp.arange(6) < n_to_roll

    def add_die(hist, i):
        die_val = rolls[i]
        should_add = roll_mask[i]
        hist = hist.at[die_val].add(jnp.where(should_add, 1, 0).astype(hist.dtype))
        return hist, None

    new_hist, _ = jax.lax.scan(add_die, new_hist.astype(jnp.int32), jnp.arange(6))
    new_hist = new_hist.astype(jnp.int8)

    return state._replace(
        dice_hist=new_hist,
        rolls_left=state.rolls_left - 1,
        current_roll_count=state.current_roll_count + 1,
        pure_hand=new_pure,
    )


# ─── Commit Move ─────────────────────────────────────────────────────

def commit_move(state: JambState, row: jnp.int32, col: jnp.int32) -> tuple:
    """Score a move and prepare for next turn. Returns (new_state, score)."""
    score = _calc_score_for_row(state.dice_hist, row)
    board = state.board.at[row, col].set(score)

    empty_cells = jnp.sum(board == -1)
    new_rolls = jnp.where(empty_cells == 1, 5, 3)
    game_over = empty_cells == 0

    new_state = JambState(
        board=board,
        dice_hist=jnp.zeros(6, dtype=jnp.int8),
        rolls_left=jnp.int32(new_rolls),
        turn_number=state.turn_number + 1,
        announced_row=jnp.int32(-1),
        pure_hand=jnp.bool_(True),
        current_roll_count=jnp.int32(0),
        game_over=game_over,
    )
    return new_state, score


# ─── Step Function ────────────────────────────────────────────────────

def step(key: jnp.ndarray, state: JambState, action: jnp.int32):
    """
    Execute one action. Returns (new_state, obs, reward, done, info).
    Handles Keep/Roll, Score, and Announce actions via lax.cond branching.
    """
    reward = jnp.float32(0.0)
    terminated = jnp.bool_(False)

    # Determine action type
    is_keep = action < NUM_KEEP_ACTIONS
    is_score = (action >= NUM_KEEP_ACTIONS) & (action < NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS)
    is_announce = (action >= NUM_KEEP_ACTIONS + NUM_SCORE_ACTIONS) & (action < TOTAL_ACTIONS)

    # ── KEEP / ROLL ──
    def do_keep(state_reward_done):
        s, r, d = state_reward_done
        pattern = KEEP_PATTERNS[action].astype(jnp.int32)  # indexing with traced value is ok for keep patterns
        k1, k2 = jax.random.split(key)
        new_s = roll_dice(k1, s, pattern)
        return new_s, r, d

    # ── SCORE ──
    def do_score(state_reward_done):
        s, r, d = state_reward_done
        score_idx = action - NUM_KEEP_ACTIONS
        row = score_idx // NUM_COLS
        col = score_idx % NUM_COLS
        new_s, move_score = commit_move(s, row, col)

        # Check game over
        is_over = new_s.game_over
        total = calculate_total_score(new_s.board)
        game_reward = jnp.where(is_over, total.astype(jnp.float32) * 0.01, 0.0)

        # Auto-roll for next turn (if not game over)
        k1, k2 = jax.random.split(key)
        rolled_s = roll_dice(k1, new_s, jnp.zeros(6, dtype=jnp.int32))
        final_s = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_over, a, b), new_s, rolled_s
        )
        return final_s, game_reward, is_over

    # ── ANNOUNCE ──
    def do_announce(state_reward_done):
        s, r, d = state_reward_done
        row = action - NUM_KEEP_ACTIONS - NUM_SCORE_ACTIONS
        new_s = s._replace(announced_row=jnp.int32(row))
        return new_s, r, d

    # ── NO-OP (safety) ──
    def do_noop(state_reward_done):
        return state_reward_done

    # Use lax.switch for dispatch
    action_type = jnp.where(is_keep, 0, jnp.where(is_score, 1, jnp.where(is_announce, 2, 3)))

    new_state, reward, terminated = jax.lax.switch(
        action_type,
        [do_keep, do_score, do_announce, do_noop],
        (state, reward, terminated),
    )

    # Stalemate check: no rolls left and no valid scoring moves
    has_rolls = new_state.rolls_left > 0
    valid_scores = get_valid_score_mask(new_state)
    has_scores = valid_scores.any()
    stalemate = (~has_rolls) & (~has_scores) & (~terminated)
    total_on_stalemate = calculate_total_score(new_state.board)
    stalemate_reward = jnp.where(stalemate, total_on_stalemate.astype(jnp.float32) * 0.01, 0.0)
    reward = reward + stalemate_reward
    terminated = terminated | stalemate

    obs = get_obs(new_state)
    info = {"score": calculate_total_score(new_state.board)}

    return new_state, obs, reward, terminated, info


# ─── Reset ────────────────────────────────────────────────────────────

def reset(key: jnp.ndarray):
    """Reset environment to initial state with first roll done."""
    state = JambState(
        board=jnp.full((NUM_ROWS, NUM_COLS), -1, dtype=jnp.int32),
        dice_hist=jnp.zeros(6, dtype=jnp.int8),
        rolls_left=jnp.int32(3),
        turn_number=jnp.int32(1),
        announced_row=jnp.int32(-1),
        pure_hand=jnp.bool_(True),
        current_roll_count=jnp.int32(0),
        game_over=jnp.bool_(False),
    )
    # Initial roll (keep nothing -> roll all 6)
    state = roll_dice(key, state, jnp.zeros(6, dtype=jnp.int32))
    obs = get_obs(state)
    return state, obs


# ─── Vectorized Env Wrapper ──────────────────────────────────────────

class JambVecEnv:
    """Vectorized environment for use with PureJaxRL-style training."""

    def __init__(self):
        self.num_actions = TOTAL_ACTIONS
        self.obs_size = OBS_SIZE

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, num_envs):
        keys = jax.random.split(key, num_envs)
        states, obs = jax.vmap(reset)(keys)
        return states, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, states, actions):
        keys = jax.random.split(key, actions.shape[0])
        return jax.vmap(step)(keys, states, actions)

    @partial(jax.jit, static_argnums=(0,))
    def get_masks(self, states):
        return jax.vmap(get_action_mask)(states)
