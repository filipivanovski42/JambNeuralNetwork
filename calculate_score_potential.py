import numpy as np

def calculate_max_possible():
    # Per Column Calculation
    
    # Section 1: Numbers (1-6)
    # Max is 5x of each number.
    sec1_max = (5*1 + 5*2 + 5*3 + 5*4 + 5*5 + 5*6) + 30 # sum + bonus
    
    # Section 2: Max-Min * 1s
    # Best case: Max=30 (5x6), Min=5 (5x1), 1s=5 (5x1)
    sec2_max = (30 - 5) * 5
    
    # Section 3: Combinations
    trips_max = (3 * 6) + 20
    straight_max = 50
    full_max = (3 * 6 + 2 * 5) + 40
    poker_max = (4 * 6) + 50
    yamb_max = (5 * 6) + 60
    sec3_max = trips_max + straight_max + full_max + poker_max + yamb_max
    
    total_per_col = sec1_max + sec2_max + sec3_max
    return {
        "sec1": sec1_max,
        "sec2": sec2_max,
        "sec3": sec3_max,
        "total_col": total_per_col,
        "grand_total": total_per_col * 4
    }

def estimate_perfect_average():
    """
    Estimates the 'Master Level' average score.
    A master player doesn't get 5-of-a-kind Every time.
    Calculations based on 3-roll probabilities (heuristic).
    """
    
    # Section 1: Numbers
    # Average 1-6 section (Master) usually hits the bonus easily.
    # Estimated average per number: ~3.8 units (e.g. 4x6, 4x5, 3x4...)
    # 3.8 * 21 = ~80. Bonus = 30. Total = 110.
    s1_avg = 110
    
    # Section 2: Max-Min * 1s
    # Max usually ~27 (mostly 5s and 6s)
    # Min usually ~7 (mostly 1s and 2s)
    # 1s usually ~3.5
    # (27 - 7) * 3.5 = 20 * 3.5 = 70
    s2_avg = 70
    
    # Section 3: Combinations
    # Trips: ~30 (avg)
    # Straight: ~35 (hit prob 70% * 50)
    # Full: ~45 (hit prob 75% * 60)
    # Poker: ~40 (hit prob 60% * 70)
    # Yamb: ~35 (hit prob 40% * 85)
    s3_avg = 30 + 35 + 45 + 40 + 35 # ~185
    
    total_avg_col = s1_avg + s2_avg + s3_avg
    return total_avg_col * 4

if __name__ == "__main__":
    max_stats = calculate_max_possible()
    avg_est = estimate_perfect_average()
    
    print("--- JAMB VARIANT THEORETICAL ANALYSIS ---")
    print(f"Absolute Maximum (Perfect Luck): {max_stats['grand_total']}")
    print(f"  - Section 1 (Numbers): {max_stats['sec1'] * 4}")
    print(f"  - Section 2 (MinMax):  {max_stats['sec2'] * 4}")
    print(f"  - Section 3 (Combs):   {max_stats['sec3'] * 4}")
    print("-" * 40)
    print(f"Estimated 'Master Player' Average: ~{int(avg_est)}")
    print(f"Current AI Performance: ~1010")
    print(f"Room for Growth: ~{int(avg_est - 1010)} points")
