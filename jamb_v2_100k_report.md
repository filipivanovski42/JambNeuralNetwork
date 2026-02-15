# Jamb Agent V2 Evaluation Report
**Games:** 100,000 | **Model:** `ckpt_997195776.npz`  
**Device:** GPU (via WSL2 JAX)

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **1705.79** |
| **Max** | **2023** |
| Median | 1714.0 |
| StdDev | 107.50 |
| Min | 1114 |

### Percentiles
| % | Score |
|---|---|
| 1% | 1423 |
| 10% | 1563 |
| 25% | 1638 |
| 50% | 1714 |
| 75% | 1783 |
| 90% | 1838 |
| 99% | 1919 |

## ⏱️ Column Completion Speed
Average turn number when the column was fully filled (Lower is faster, but usually constrained by rules).
For 'Up' column, it fills bottom-to-top, so 'faster' means finishing 1s earlier.
Wait, game ends around turn 50-60.

| Column | Avg Turn Filled |
|:---|:---|
| **Down** | 45.3 |
| **Free** | 48.4 |
| **Up** | 50.3 |
| **Anno** | 50.4 |

## 🎲 Average Board Values
(Averaged across 100k games)

| Row | Down | Free | Up | Anno |
|:----|:---:|:---:|:---:|:---:|
| **1s** | 3.51 | 4.03 | 3.37 | 3.88 |
| **2s** | 5.16 | 4.53 | 4.79 | 3.41 |
| **3s** | 8.63 | 8.78 | 8.03 | 7.62 |
| **4s** | 12.19 | 12.87 | 11.32 | 12.95 |
| **5s** | 15.20 | 16.71 | 14.75 | 17.00 |
| **6s** | 18.91 | 20.83 | 18.60 | 20.48 |
| **Max** | 26.14 | 26.26 | 25.81 | 26.07 |
| **Min** | 8.71 | 8.32 | 8.66 | 8.22 |
| **T** | 34.64 | 35.60 | 34.68 | 32.64 |
| **K** | 48.49 | 48.46 | 48.06 | 48.57 |
| **F** | 63.24 | 63.68 | 63.55 | 61.01 |
| **P** | 68.01 | 71.10 | 68.98 | 62.01 |
| **Y** | 58.48 | 79.50 | 73.37 | 38.20 |

## 📜 Best Game Log (Score: 2023)
Seed: `636247987`

```text
--- Replaying Game with Seed 636247987 ---

⚡ TURN 1 (Rolls: 2)
🎲 Dice: [2, 3, 5, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 1 (Rolls: 1)
🎲 Dice: [1, 2, 3, 3, 6, 6]
👉 KEEP: nothing

⚡ TURN 1 (Rolls: 0)
🎲 Dice: [2, 4, 5, 6, 6, 6]
� SCORE: T in Anno
   Current Score: 38

⚡ TURN 2 (Rolls: 2)
🎲 Dice: [2, 2, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 2 (Rolls: 1)
🎲 Dice: [3, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 2 (Rolls: 0)
🎲 Dice: [5, 6, 6, 6, 6, 6]
� SCORE: Y in Up
   Current Score: 128

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [1, 1, 4, 5, 6, 6]
👉 KEEP: [1, 1]

⚡ TURN 3 (Rolls: 1)
🎲 Dice: [1, 1, 4, 5, 5, 6]
👉 KEEP: [1, 1]

⚡ TURN 3 (Rolls: 0)
🎲 Dice: [1, 1, 2, 3, 6, 6]
� SCORE: 2s in Free
   Current Score: 130

⚡ TURN 4 (Rolls: 2)
🎲 Dice: [2, 2, 2, 3, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 4 (Rolls: 1)
🎲 Dice: [5, 5, 5, 6, 6, 6]
👉 KEEP: [5, 5, 6, 6, 6]

⚡ TURN 4 (Rolls: 0)
🎲 Dice: [5, 5, 6, 6, 6, 6]
� SCORE: P in Up
   Current Score: 204

⚡ TURN 5 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 3, 6]
👉 KEEP: [1, 1]

⚡ TURN 5 (Rolls: 1)
🎲 Dice: [1, 1, 1, 3, 5, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 5 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 2]
� SCORE: 1s in Down
   Current Score: 209

⚡ TURN 6 (Rolls: 2)
🎲 Dice: [1, 1, 5, 5, 5, 5]
� ANNOUNCE: P

⚡ TURN 6 (Rolls: 2)
🎲 Dice: [1, 1, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 6 (Rolls: 1)
🎲 Dice: [2, 5, 5, 5, 5, 5]
� SCORE: P in Anno
   Current Score: 279

⚡ TURN 7 (Rolls: 2)
🎲 Dice: [1, 3, 3, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 7 (Rolls: 1)
🎲 Dice: [3, 3, 5, 6, 6, 6]
👉 KEEP: [3, 3, 6, 6, 6]

⚡ TURN 7 (Rolls: 0)
🎲 Dice: [3, 3, 5, 6, 6, 6]
� SCORE: F in Up
   Current Score: 343

⚡ TURN 8 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 5, 5]
� ANNOUNCE: 5s

⚡ TURN 8 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 8 (Rolls: 1)
🎲 Dice: [1, 2, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 8 (Rolls: 0)
🎲 Dice: [2, 5, 5, 5, 5, 5]
� SCORE: 5s in Anno
   Current Score: 368

⚡ TURN 9 (Rolls: 2)
🎲 Dice: [3, 3, 3, 3, 4, 5]
� ANNOUNCE: Y

⚡ TURN 9 (Rolls: 2)
🎲 Dice: [3, 3, 3, 3, 4, 5]
👉 KEEP: [3, 3, 3, 3]

⚡ TURN 9 (Rolls: 1)
🎲 Dice: [3, 3, 3, 3, 3, 5]
👉 KEEP: [3, 3, 3, 3, 3]

⚡ TURN 9 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 3, 3]
� SCORE: Y in Anno
   Current Score: 443

⚡ TURN 10 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 6]
👉 KEEP: [2, 3, 4, 6]

⚡ TURN 10 (Rolls: 1)
🎲 Dice: [1, 2, 3, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 10 (Rolls: 0)
🎲 Dice: [2, 3, 4, 5, 5, 6]
� SCORE: K in Up
   Current Score: 493

⚡ TURN 11 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 4, 6]
👉 KEEP: [2, 2]

⚡ TURN 11 (Rolls: 1)
🎲 Dice: [2, 2, 3, 5, 5, 6]
👉 KEEP: [2, 2, 3]

⚡ TURN 11 (Rolls: 0)
🎲 Dice: [1, 2, 2, 2, 3, 6]
� SCORE: 2s in Down
   Current Score: 499

⚡ TURN 12 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [4, 6, 6]

⚡ TURN 12 (Rolls: 1)
🎲 Dice: [1, 3, 3, 4, 6, 6]
👉 KEEP: [3, 3]

⚡ TURN 12 (Rolls: 0)
🎲 Dice: [2, 2, 3, 3, 4, 5]
� SCORE: 3s in Down
   Current Score: 505

⚡ TURN 13 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 13 (Rolls: 1)
🎲 Dice: [2, 4, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 13 (Rolls: 0)
🎲 Dice: [1, 3, 6, 6, 6, 6]
� SCORE: P in Free
   Current Score: 579

⚡ TURN 14 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 5, 6]
👉 KEEP: [5, 6]

⚡ TURN 14 (Rolls: 1)
🎲 Dice: [1, 4, 4, 5, 6, 6]
👉 KEEP: [4, 4]

⚡ TURN 14 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 4, 5]
� SCORE: 4s in Down
   Current Score: 595

⚡ TURN 15 (Rolls: 2)
🎲 Dice: [2, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 15 (Rolls: 1)
🎲 Dice: [2, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 15 (Rolls: 0)
🎲 Dice: [5, 6, 6, 6, 6, 6]
� SCORE: Y in Free
   Current Score: 685

⚡ TURN 16 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 2, 2]
� ANNOUNCE: 1s

⚡ TURN 16 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 2, 2]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 16 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 2, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 16 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 2]
� SCORE: 1s in Anno
   Current Score: 690

⚡ TURN 17 (Rolls: 2)
🎲 Dice: [1, 1, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 17 (Rolls: 1)
🎲 Dice: [2, 2, 5, 5, 6, 6]
👉 KEEP: [5, 5, 6, 6]

⚡ TURN 17 (Rolls: 0)
🎲 Dice: [1, 4, 5, 5, 6, 6]
� SCORE: 5s in Down
   Current Score: 700

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 2, 2, 5, 5, 5]
👉 KEEP: [1, 5, 5, 5]

⚡ TURN 18 (Rolls: 1)
🎲 Dice: [1, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5, 6]

⚡ TURN 18 (Rolls: 0)
🎲 Dice: [4, 4, 5, 5, 5, 6]
� SCORE: T in Up
   Current Score: 735

⚡ TURN 19 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 19 (Rolls: 1)
🎲 Dice: [1, 1, 1, 5, 6, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 19 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 4, 6]
� SCORE: Min in Up
   Current Score: 735

⚡ TURN 20 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 3, 6]
👉 KEEP: [6]

⚡ TURN 20 (Rolls: 1)
🎲 Dice: [3, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 20 (Rolls: 0)
🎲 Dice: [5, 6, 6, 6, 6, 6]
� SCORE: 6s in Down
   Current Score: 795

⚡ TURN 21 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 6]
👉 KEEP: [6]

⚡ TURN 21 (Rolls: 1)
🎲 Dice: [2, 3, 3, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 21 (Rolls: 0)
🎲 Dice: [1, 4, 5, 6, 6, 6]
� SCORE: Max in Down
   Current Score: 795

⚡ TURN 22 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 5, 6]
👉 KEEP: [1, 1]

⚡ TURN 22 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 2, 6]
👉 KEEP: [1, 1, 2, 2, 2]

⚡ TURN 22 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 2, 6]
� SCORE: Min in Down
   Current Score: 890

⚡ TURN 23 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 4, 6]
👉 KEEP: [6]

⚡ TURN 23 (Rolls: 1)
🎲 Dice: [1, 1, 2, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 23 (Rolls: 0)
🎲 Dice: [1, 2, 4, 6, 6, 6]
� SCORE: T in Down
   Current Score: 928

⚡ TURN 24 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 5]
👉 KEEP: [1, 2, 3, 4, 5]

⚡ TURN 24 (Rolls: 1)
🎲 Dice: [1, 2, 2, 3, 4, 5]
👉 KEEP: [1, 2, 3, 4, 5]

⚡ TURN 24 (Rolls: 0)
🎲 Dice: [1, 2, 3, 3, 4, 5]
� SCORE: K in Down
   Current Score: 973

⚡ TURN 25 (Rolls: 2)
🎲 Dice: [2, 4, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 25 (Rolls: 1)
🎲 Dice: [2, 5, 5, 6, 6, 6]
👉 KEEP: [5, 5, 6, 6, 6]

⚡ TURN 25 (Rolls: 0)
🎲 Dice: [1, 5, 5, 6, 6, 6]
� SCORE: F in Down
   Current Score: 1041

⚡ TURN 26 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 5, 6]
👉 KEEP: [5, 6]

⚡ TURN 26 (Rolls: 1)
🎲 Dice: [3, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5, 6]

⚡ TURN 26 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 5, 6]
� SCORE: P in Down
   Current Score: 1111

⚡ TURN 27 (Rolls: 2)
🎲 Dice: [2, 4, 4, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 27 (Rolls: 1)
🎲 Dice: [3, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5, 6]

⚡ TURN 27 (Rolls: 0)
🎲 Dice: [5, 5, 5, 5, 6, 6]
� SCORE: F in Free
   Current Score: 1178

⚡ TURN 28 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 3, 4]
👉 KEEP: [1, 1, 1]

⚡ TURN 28 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 3, 5]
👉 KEEP: [1, 1, 1]

⚡ TURN 28 (Rolls: 0)
🎲 Dice: [1, 1, 1, 5, 5, 6]
� SCORE: 5s in Free
   Current Score: 1188

⚡ TURN 29 (Rolls: 2)
🎲 Dice: [3, 4, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 29 (Rolls: 1)
🎲 Dice: [2, 2, 2, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 29 (Rolls: 0)
🎲 Dice: [3, 5, 5, 6, 6, 6]
� SCORE: Max in Up
   Current Score: 1188

⚡ TURN 30 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 3, 3]
👉 KEEP: [1, 1]

⚡ TURN 30 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 4, 5]
👉 KEEP: [1, 1, 1]

⚡ TURN 30 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 3, 5]
� SCORE: 1s in Free
   Current Score: 1192

⚡ TURN 31 (Rolls: 2)
🎲 Dice: [2, 2, 2, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 31 (Rolls: 1)
🎲 Dice: [2, 4, 4, 5, 5, 6]
👉 KEEP: nothing

⚡ TURN 31 (Rolls: 0)
🎲 Dice: [3, 4, 4, 5, 5, 5]
� SCORE: F in Anno
   Current Score: 1255

⚡ TURN 32 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 32 (Rolls: 1)
🎲 Dice: [1, 2, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 32 (Rolls: 0)
🎲 Dice: [4, 5, 6, 6, 6, 6]
� SCORE: 6s in Up
   Current Score: 1279

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 3, 4]
� ANNOUNCE: 3s

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 3, 4]
👉 KEEP: [3, 3, 3]

⚡ TURN 33 (Rolls: 1)
🎲 Dice: [1, 2, 3, 3, 3, 5]
👉 KEEP: [1, 3, 3, 3, 5]

⚡ TURN 33 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 5, 6]
� SCORE: 3s in Anno
   Current Score: 1288

⚡ TURN 34 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 34 (Rolls: 1)
🎲 Dice: [1, 2, 2, 2, 6, 6]
👉 KEEP: [1, 2, 2, 2]

⚡ TURN 34 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 2, 2]
� SCORE: Min in Free
   Current Score: 1288

⚡ TURN 35 (Rolls: 2)
🎲 Dice: [1, 1, 5, 5, 6, 6]
👉 KEEP: [5, 5]

⚡ TURN 35 (Rolls: 1)
🎲 Dice: [1, 5, 5, 5, 5, 5]
� SCORE: Y in Down
   Current Score: 1373

⚡ TURN 36 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 36 (Rolls: 1)
🎲 Dice: [5, 5, 5, 5, 5, 6]
� SCORE: 5s in Up
   Current Score: 1398

⚡ TURN 37 (Rolls: 2)
🎲 Dice: [1, 3, 3, 3, 5, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 37 (Rolls: 1)
🎲 Dice: [3, 3, 3, 3, 4, 6]
👉 KEEP: [3, 3, 3, 3]

⚡ TURN 37 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 3, 6]
� SCORE: 3s in Free
   Current Score: 1410

⚡ TURN 38 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 6]
👉 KEEP: [4, 4]

⚡ TURN 38 (Rolls: 1)
🎲 Dice: [4, 4, 5, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6]

⚡ TURN 38 (Rolls: 0)
🎲 Dice: [2, 4, 5, 6, 6, 6]
� SCORE: T in Free
   Current Score: 1448

⚡ TURN 39 (Rolls: 2)
🎲 Dice: [1, 4, 4, 5, 5, 5]
👉 KEEP: [4, 4]

⚡ TURN 39 (Rolls: 1)
🎲 Dice: [1, 2, 2, 3, 4, 4]
👉 KEEP: [4, 4]

⚡ TURN 39 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 4, 4]
� SCORE: 4s in Up
   Current Score: 1494

⚡ TURN 40 (Rolls: 2)
🎲 Dice: [3, 4, 5, 6, 6, 6]
� ANNOUNCE: 6s

⚡ TURN 40 (Rolls: 2)
🎲 Dice: [3, 4, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 40 (Rolls: 1)
🎲 Dice: [1, 4, 4, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 40 (Rolls: 0)
🎲 Dice: [2, 4, 6, 6, 6, 6]
� SCORE: 6s in Anno
   Current Score: 1548

⚡ TURN 41 (Rolls: 2)
🎲 Dice: [2, 2, 2, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 41 (Rolls: 1)
🎲 Dice: [2, 4, 5, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6]

⚡ TURN 41 (Rolls: 0)
🎲 Dice: [1, 5, 6, 6, 6, 6]
� SCORE: 6s in Free
   Current Score: 1572

⚡ TURN 42 (Rolls: 2)
🎲 Dice: [1, 3, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 42 (Rolls: 1)
🎲 Dice: [5, 6, 6, 6, 6, 6]
� SCORE: Max in Free
   Current Score: 1660

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 6]
� ANNOUNCE: 4s

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 6]
👉 KEEP: [4, 4]

⚡ TURN 43 (Rolls: 1)
🎲 Dice: [1, 3, 4, 4, 6, 6]
👉 KEEP: [4, 4]

⚡ TURN 43 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 6, 6]
� SCORE: 4s in Anno
   Current Score: 1668

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [3, 4, 5]

⚡ TURN 44 (Rolls: 1)
🎲 Dice: [1, 1, 3, 4, 4, 5]
👉 KEEP: nothing

⚡ TURN 44 (Rolls: 0)
🎲 Dice: [2, 3, 4, 5, 6, 6]
� SCORE: K in Anno
   Current Score: 1718

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [1, 2, 2, 2, 3, 4]
� SCORE: 2s in Anno
   Current Score: 1724

⚡ TURN 46 (Rolls: 2)
🎲 Dice: [2, 2, 2, 3, 4, 6]
👉 KEEP: [2, 3, 4, 6]

⚡ TURN 46 (Rolls: 1)
🎲 Dice: [2, 3, 4, 4, 4, 6]
👉 KEEP: [4, 4, 4]

⚡ TURN 46 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 4, 5]
� SCORE: 4s in Free
   Current Score: 1766

⚡ TURN 47 (Rolls: 2)
🎲 Dice: [1, 3, 3, 4, 5, 6]
👉 KEEP: [3, 3]

⚡ TURN 47 (Rolls: 1)
🎲 Dice: [1, 1, 3, 3, 3, 5]
👉 KEEP: [3, 3, 3]

⚡ TURN 47 (Rolls: 0)
🎲 Dice: [3, 3, 3, 5, 5, 5]
� SCORE: 3s in Up
   Current Score: 1775

⚡ TURN 48 (Rolls: 2)
🎲 Dice: [2, 3, 3, 4, 5, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 48 (Rolls: 1)
🎲 Dice: [2, 2, 3, 3, 4, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 48 (Rolls: 0)
🎲 Dice: [1, 2, 2, 3, 4, 5]
� SCORE: K in Free
   Current Score: 1820

⚡ TURN 49 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 5, 5]
� ANNOUNCE: Min

⚡ TURN 49 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 5, 5]
👉 KEEP: [1, 1, 2, 2]

⚡ TURN 49 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 4, 5]
👉 KEEP: [1, 1, 2, 2]

⚡ TURN 49 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 3, 4]
� SCORE: Min in Anno
   Current Score: 1820

⚡ TURN 50 (Rolls: 2)
🎲 Dice: [2, 2, 2, 4, 5, 5]
👉 KEEP: [2, 2, 2, 4]

⚡ TURN 50 (Rolls: 1)
🎲 Dice: [2, 2, 2, 2, 3, 4]
👉 KEEP: [2, 2, 2, 2]

⚡ TURN 50 (Rolls: 0)
🎲 Dice: [1, 2, 2, 2, 2, 6]
� SCORE: 2s in Up
   Current Score: 1828

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [3, 4, 4, 5, 6, 6]
� ANNOUNCE: Max

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [3, 4, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 51 (Rolls: 1)
🎲 Dice: [1, 4, 5, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6]

⚡ TURN 51 (Rolls: 0)
🎲 Dice: [2, 4, 5, 6, 6, 6]
� SCORE: Max in Anno
   Current Score: 1918

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [1, 1, 2, 3, 5, 6]
👉 KEEP: [1, 1]

⚡ TURN 52 (Rolls: 3)
🎲 Dice: [1, 1, 1, 1, 1, 4]
👉 KEEP: [1, 1, 1, 1, 1]

⚡ TURN 52 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 1, 5]
� SCORE: 1s in Up
   Current Score: 2023

        Down     Free      Up      Anno  
      ------------------------------------
  1s |    5        4        5        5     
  2s |    6        2        8        6     
  3s |    6        12       9        9     
  4s |    16       12       16       8     
  5s |    10       10       25       25    
  6s |    30       24       24       24    
 Max |    27       30       28       27    
 Min |    8        8        8        9     
   T |    38       38       35       38    
   K |    45       45       50       50    
   F |    68       67       64       63    
   P |    70       74       74       70    
   Y |    85       90       90       75    
      ------------------------------------
🏁 FINAL SCORE: 2023

```
