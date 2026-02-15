import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from jamb_logic import JambLogic, NUM_ROWS, NUM_COLS, NUM_DICE, Row, Column


class JambEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    # Action layout constants (set in __init__ after loading patterns)
    NUM_KEEP_ACTIONS = 0
    NUM_SCORE_ACTIONS = NUM_ROWS * NUM_COLS  # 52
    NUM_ANNOUNCE_ACTIONS = NUM_ROWS           # 13

    def __init__(self):
        super().__init__()
        self.logic = JambLogic()

        # ── Load precomputed tables ──────────────────────────────
        base = os.path.dirname(os.path.abspath(__file__))
        kp_path = os.path.join(base, "keep_patterns.npy")
        es_path = os.path.join(base, "expected_scores.npy")

        if not os.path.exists(kp_path) or not os.path.exists(es_path):
            raise FileNotFoundError(
                "Precomputed tables not found. Run `python precompute_tables.py` first."
            )

        all_keep_patterns = np.load(kp_path)        # (924, 6)
        all_expected_scores = np.load(es_path)       # (924, 6, 13)

        # ── V6 FIX: Remove keep-all-6 patterns ──────────────────
        # Keeping all 6 dice = rolling 0 dice = invalid move.
        # A roll must reroll at least 1 die.
        valid = all_keep_patterns.sum(axis=1) < 6
        self.keep_patterns = all_keep_patterns[valid]              # (462, 6)
        self.expected_scores_table = all_expected_scores[valid]    # (462, 6, 13)

        # Build histogram-tuple → pattern index lookup
        self.hist_to_idx = {}
        for i, pat in enumerate(self.keep_patterns):
            self.hist_to_idx[tuple(pat)] = i

        self.NUM_KEEP_ACTIONS = len(self.keep_patterns)  # 462

        # ── Action Space ─────────────────────────────────────────
        # 0 .. 461          : Keep patterns (must reroll ≥1 die)
        # 462 .. 513        : Score (row*4 + col)
        # 514 .. 526        : Announce (row 0-12)
        total_actions = self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS + self.NUM_ANNOUNCE_ACTIONS
        self.action_space = spaces.Discrete(total_actions)  # 527

        # ── Observation Space (v6) ───────────────────────────────
        # Board Scores        : 52  (normalised /100)
        # Board Filled        : 52  (binary)
        # Dice Histogram      :  6  (counts[1..6] / 6)
        # Rolls Left          :  1
        # Announced Row       :  1
        # Pure Hand           :  1
        # Turn Number         :  1
        # Cell Scores         : 52  (what you'd score in each legal cell, /100)
        # ─────────────────────────
        # Total               : 166
        self.observation_space = spaces.Box(low=0, high=1, shape=(166,), dtype=np.float32)

    # ─── Reset ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.logic.reset()
        # Initial roll (keep nothing → roll all 6)
        self.logic.roll_dice(np.zeros(6, dtype=int))
        return self._get_obs(), {}

    # ─── Observation ─────────────────────────────────────────────

    def _get_obs(self):
        board_flat = self.logic.board.flatten()

        # Board scores (normalised)
        board_scores = np.copy(board_flat).astype(np.float32)
        board_scores[board_scores == -1] = 0
        board_scores /= 100.0

        # Board filled (binary)
        board_filled = (board_flat != -1).astype(np.float32)

        # Dice histogram (6 values, each /6)
        dice_hist = self.logic.dice_histogram.astype(np.float32) / 6.0

        # Scalars
        rolls_norm = self.logic.rolls_left / 5.0
        anno_norm = 0.0
        if self.logic.announced_row != -1:
            anno_norm = (self.logic.announced_row + 1) / 13.0
        turn_norm = self.logic.turn_number / 52.0
        pure_hand = 1.0 if self.logic.pure_hand else 0.0

        # ── V6: Per-cell scores for legal moves ──────────────────
        # For each of the 52 board cells, show the score you'd get
        # if you scored there RIGHT NOW. 0 for non-legal moves.
        # This directly encodes: cell availability + column constraints
        # + announcement lock + current hand value.
        cell_scores = np.zeros(NUM_ROWS * NUM_COLS, dtype=np.float32)
        sorted_dice = self.logic.get_sorted_dice()
        valid_moves = self.logic.get_available_actions()
        for (r, c) in valid_moves:
            score = self.logic.calculate_score_for_move(r, c, sorted_dice)
            cell_scores[r * NUM_COLS + c] = score / 100.0

        obs = np.concatenate([
            board_scores,       # 52
            board_filled,       # 52
            dice_hist,          # 6
            [rolls_norm],       # 1
            [anno_norm],        # 1
            [pure_hand],        # 1
            [turn_norm],        # 1
            cell_scores,        # 52
        ]).astype(np.float32)
        return obs

    # ─── Step ────────────────────────────────────────────────────

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # ── KEEP / ROLL (0 .. NUM_KEEP_ACTIONS-1) ───────────────
        if action < self.NUM_KEEP_ACTIONS:
            if self.logic.rolls_left > 0:
                pattern = self.keep_patterns[action].astype(int)
                self.logic.roll_dice(pattern)
            else:
                reward = -1.0  # safety net (masks should prevent this)

        # ── SCORE (NUM_KEEP_ACTIONS .. NUM_KEEP_ACTIONS+51) ──────
        elif action < self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS:
            score_idx = action - self.NUM_KEEP_ACTIONS
            r = score_idx // NUM_COLS
            c = score_idx % NUM_COLS

            valid_moves = self.logic.get_available_actions()
            if (r, c) in valid_moves:
                self.logic.commit_move(r, c)

                if self.logic.game_over:
                    terminated = True
                    reward = self.logic.calculate_total_score() * 0.01
                else:
                    # Auto-roll for next turn start
                    self.logic.roll_dice(np.zeros(6, dtype=int))
            else:
                reward = -1.0

        # ── ANNOUNCE (NUM_KEEP_ACTIONS+52 .. NUM_KEEP_ACTIONS+64) ─
        elif action < self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS + self.NUM_ANNOUNCE_ACTIONS:
            row = action - self.NUM_KEEP_ACTIONS - self.NUM_SCORE_ACTIONS
            if self.logic.current_roll_count == 1 and self.logic.announced_row == -1:
                if self.logic.board[row, Column.ANNOUNCE] == -1:
                    self.logic.announce(row)
                else:
                    reward = -1.0
            else:
                reward = -1.0
        else:
            reward = -1.0

        # ── Stalemate check ──────────────────────────────────────
        if not terminated:
            if self.logic.rolls_left <= 0:
                valid_scores = self.logic.get_available_actions()
                if len(valid_scores) == 0:
                    terminated = True
                    # Still give the final score as reward
                    reward = self.logic.calculate_total_score() * 0.01

        return self._get_obs(), reward, terminated, truncated, info

    # ─── Action Masks ────────────────────────────────────────────

    def action_masks(self):
        total = self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS + self.NUM_ANNOUNCE_ACTIONS
        mask = np.zeros(total, dtype=bool)

        hist = self.logic.dice_histogram

        # ── Keep actions (0..461) — vectorised ────────────────────
        # All patterns already exclude keep-all-6 (filtered in __init__)
        if self.logic.rolls_left > 0:
            valid_keeps = np.all(self.keep_patterns <= hist, axis=1)  # (462,)
            mask[:self.NUM_KEEP_ACTIONS] = valid_keeps

        # ── Score actions (462..513) ─────────────────────────────
        valid_moves = self.logic.get_available_actions()
        for (r, c) in valid_moves:
            idx = self.NUM_KEEP_ACTIONS + r * NUM_COLS + c
            mask[idx] = True

        # ── Announce actions (514..526) ──────────────────────────
        if self.logic.current_roll_count == 1 and self.logic.announced_row == -1:
            for r in range(NUM_ROWS):
                if self.logic.board[r, Column.ANNOUNCE] == -1:
                    mask[self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS + r] = True

        # ── Force announcement if all remaining cells are Announce ──
        empty_cells = []
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if self.logic.board[r, c] == -1:
                    empty_cells.append((r, c))

        if len(empty_cells) > 0:
            all_in_announce = all(c == Column.ANNOUNCE for _, c in empty_cells)
            if (all_in_announce 
                and self.logic.current_roll_count == 1 
                and self.logic.announced_row == -1):
                # Force: ONLY announce actions available
                mask[:] = False
                for r, c in empty_cells:
                    mask[self.NUM_KEEP_ACTIONS + self.NUM_SCORE_ACTIONS + r] = True

        # ── Safety fallback ──────────────────────────────────────
        if not np.any(mask):
            mask[:] = True

        return mask
