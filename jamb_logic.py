import numpy as np
from enum import IntEnum

class Column(IntEnum):
    DOWN = 0
    FREE = 1
    UP = 2
    ANNOUNCE = 3

class Row(IntEnum):
    ONES = 0
    TWOS = 1
    THREES = 2
    FOURS = 3
    FIVES = 4
    SIXES = 5
    MAX = 6
    MIN = 7
    TRIPS = 8
    STRAIGHT = 9
    FULL = 10
    POKER = 11
    YAMB = 12

NUM_ROWS = 13
NUM_COLS = 4
NUM_DICE = 6

class JambLogic:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.full((NUM_ROWS, NUM_COLS), -1, dtype=int)
        self.dice_histogram = np.zeros(6, dtype=int)  # counts of [1s, 2s, 3s, 4s, 5s, 6s]
        
        # Turn tracking
        self.rolls_left = 3
        self.turn_number = 1 
        self.announced_row = -1 
        self.pure_hand = True 
        self.current_roll_count = 0
        self.game_over = False

    def get_sorted_dice(self):
        """Expand histogram to a sorted array of 6 dice values."""
        dice = []
        for val_idx in range(6):
            dice.extend([val_idx + 1] * int(self.dice_histogram[val_idx]))
        return np.array(dice, dtype=int)

    def roll_dice(self, keep_histogram):
        """Roll dice, keeping the specified histogram of dice values.
        
        keep_histogram: array of 6 ints — how many of each value (1-6) to keep.
        Must satisfy: keep_histogram[i] <= dice_histogram[i] for all i,
                       and sum(keep_histogram) <= 6.
        """
        if self.rolls_left <= 0:
            return False
            
        n_kept = int(np.sum(keep_histogram))
        
        # Pure Hand logic
        if self.current_roll_count > 0:
            if n_kept > 0:
                self.pure_hand = False
            else:
                self.pure_hand = True
        else:
            # First roll: always pure hand
            self.pure_hand = True

        # Start with kept dice
        self.dice_histogram = np.array(keep_histogram, dtype=int)
        
        # Roll remaining dice
        n_to_roll = 6 - n_kept
        for _ in range(n_to_roll):
            val = np.random.randint(1, 7)
            self.dice_histogram[val - 1] += 1
        
        self.rolls_left -= 1
        self.current_roll_count += 1
        return True

    def announce(self, row):
        """Announce a row in Column E. Valid ONLY after first roll."""
        if self.current_roll_count != 1:
            return False
        if self.announced_row != -1:
            return False
        if self.board[row, Column.ANNOUNCE] != -1:
            return False
        self.announced_row = row
        return True

    def get_available_actions(self):
        """Returns list of valid (row, col) scoring moves."""
        valid_cells = []
        is_empty = lambda r, c: self.board[r, c] == -1
        
        for c in range(NUM_COLS):
            for r in range(NUM_ROWS):
                if not is_empty(r, c):
                    continue
                
                if c == Column.DOWN:
                    if r == 0 or not is_empty(r - 1, c):
                        valid_cells.append((r, c))
                elif c == Column.FREE:
                    valid_cells.append((r, c))
                elif c == Column.UP:
                    if r == NUM_ROWS - 1 or not is_empty(r + 1, c):
                        valid_cells.append((r, c))
                elif c == Column.ANNOUNCE:
                    if self.announced_row != -1:
                        if r == self.announced_row:
                            valid_cells.append((r, c))
                    elif self.pure_hand:
                        valid_cells.append((r, c))

        # Enforce Announcement Constraint: if announced, MUST score there
        if self.announced_row != -1:
            valid_cells = [(r, c) for r, c in valid_cells 
                          if c == Column.ANNOUNCE and r == self.announced_row]

        return valid_cells

    def calculate_score_for_move(self, row, col, dice):
        """Calculate score for a move. `dice` is a sorted array of 6 values."""
        sorted_dice = np.sort(dice)
        counts = np.bincount(sorted_dice, minlength=7)
        score = 0
        
        if row <= Row.SIXES:
            target_val = row + 1
            num_matches = min(counts[target_val], 5)
            score = num_matches * target_val
            
        elif row == Row.MAX:
            score = np.sum(sorted_dice[-5:])
        elif row == Row.MIN:
            score = np.sum(sorted_dice[:5])
            
        elif row == Row.TRIPS:
            val = 0
            for v in range(6, 0, -1):
                if counts[v] >= 3:
                    val = v
                    break
            if val > 0: score = (3 * val) + 20
                
        elif row == Row.STRAIGHT:
            unique = np.unique(sorted_dice)
            has_small = np.all(np.isin([1,2,3,4,5], unique))
            has_large = np.all(np.isin([2,3,4,5,6], unique))
            if has_large: score = 50
            elif has_small: score = 45
            
        elif row == Row.FULL:
            possible = [0]
            for v1 in range(6, 0, -1):
                if counts[v1] >= 3:
                    for v2 in range(6, 0, -1):
                         if v1 == v2: continue
                         if counts[v2] >= 2:
                             possible.append((3*v1 + 2*v2) + 40)
            score = max(possible)

        elif row == Row.POKER:
            val = 0
            for v in range(6, 0, -1):
                if counts[v] >= 4:
                    val = v
                    break
            if val > 0: score = (4 * val) + 50

        elif row == Row.YAMB:
            val = 0
            for v in range(6, 0, -1):
                if counts[v] >= 5:
                    val = v
                    break
            if val > 0: score = (5 * val) + 60

        return score

    def commit_move(self, row, col):
        sorted_dice = self.get_sorted_dice()
        score = self.calculate_score_for_move(row, col, sorted_dice)
        self.board[row, col] = score
        
        # Prepare for next turn
        self.dice_histogram = np.zeros(6, dtype=int)
        
        empty_cells = np.sum(self.board == -1)
        if empty_cells == 1:
            self.rolls_left = 5
        else:
            self.rolls_left = 3
            
        self.current_roll_count = 0
        self.announced_row = -1
        self.pure_hand = True
        self.turn_number += 1
        
        if empty_cells == 0:
            self.game_over = True
            
        return score

    def calculate_total_score(self):
        total = 0
        for c in range(NUM_COLS):
            col_score = 0
            
            # Sec 1: Numbers
            sum_1_6 = 0
            for r in range(6):
                val = self.board[r, c]
                if val > -1: sum_1_6 += val
            if sum_1_6 >= 60: sum_1_6 += 30
            col_score += sum_1_6
            
            # Sec 2: Max-Min
            val_max = self.board[Row.MAX, c]
            val_min = self.board[Row.MIN, c]
            val_ones = self.board[Row.ONES, c]
            
            if val_max > -1 and val_min > -1 and val_ones > -1:
                diff = val_max - val_min
                if diff < 0: diff = 0
                section2_score = diff * val_ones
                col_score += section2_score
            
            # Sec 3: Combos
            for r in range(Row.TRIPS, Row.YAMB + 1):
                val = self.board[r, c]
                if val > -1: col_score += val
            
            total += col_score
        return total
