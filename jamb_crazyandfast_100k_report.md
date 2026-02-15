# Jamb Agent (Crazy and Fast) Evaluation Report
**Games:** 100,000 | **Model:** `ckpt_3985637376.npz`  
**Device:** GPU (via WSL2 JAX)

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **1722.09** |
| **Max** | **2044** |
| Median | 1730.0 |
| StdDev | 107.02 |
| Min | 1218 |

### Percentiles
| % | Score |
|---|---|
| 1% | 1438 |
| 10% | 1581 |
| 25% | 1654 |
| 50% | 1730 |
| 75% | 1799 |
| 90% | 1853 |
| 99% | 1931 |

## ⏱️ Column Completion Speed
Average turn number when the column was fully filled (Lower is faster, but usually constrained by rules).
For 'Up' column, it fills bottom-to-top, so 'faster' means finishing 1s earlier.
Wait, game ends around turn 50-60.

| Column | Avg Turn Filled |
|:---|:---|
| **Down** | 46.2 |
| **Free** | 48.8 |
| **Up** | 50.0 |
| **Anno** | 50.1 |

## 🎲 Average Board Values
(Averaged across 100k games)

| Row | Down | Free | Up | Anno |
|:----|:---:|:---:|:---:|:---:|
| **1s** | 3.64 | 4.06 | 3.36 | 3.89 |
| **2s** | 5.29 | 4.22 | 4.81 | 3.44 |
| **3s** | 8.80 | 8.82 | 8.04 | 8.50 |
| **4s** | 12.24 | 13.01 | 11.41 | 13.04 |
| **5s** | 15.46 | 16.85 | 14.96 | 17.01 |
| **6s** | 19.76 | 20.80 | 18.79 | 20.14 |
| **Max** | 26.26 | 26.30 | 26.01 | 26.20 |
| **Min** | 8.48 | 8.13 | 8.52 | 8.12 |
| **T** | 34.67 | 35.86 | 34.49 | 33.06 |
| **K** | 48.51 | 48.42 | 47.98 | 48.64 |
| **F** | 63.41 | 63.80 | 63.56 | 61.27 |
| **P** | 67.45 | 71.16 | 68.86 | 61.88 |
| **Y** | 56.56 | 79.32 | 76.14 | 38.62 |

## 📜 Best Game Log (Score: 2044)
Seed: `433856501`

```text
--- Replaying Game with Seed 433856501 ---

⚡ TURN 1 (Rolls: 2)
🎲 Dice: [3, 4, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 1 (Rolls: 1)
🎲 Dice: [4, 4, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4, 4, 4]

⚡ TURN 1 (Rolls: 0)
🎲 Dice: [2, 4, 4, 4, 4, 4]
� SCORE: Y in Up
   Current Score: 80

⚡ TURN 2 (Rolls: 2)
🎲 Dice: [2, 2, 4, 4, 5, 6]
👉 KEEP: [4, 4]

⚡ TURN 2 (Rolls: 1)
🎲 Dice: [1, 3, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4]

⚡ TURN 2 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 4, 4]
� SCORE: P in Up
   Current Score: 146

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 6]
� ANNOUNCE: K

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 3 (Rolls: 1)
🎲 Dice: [2, 3, 3, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 3 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 5, 6]
� SCORE: K in Anno
   Current Score: 196

⚡ TURN 4 (Rolls: 2)
🎲 Dice: [1, 2, 5, 5, 5, 5]
� SCORE: P in Anno
   Current Score: 266

⚡ TURN 5 (Rolls: 2)
🎲 Dice: [1, 1, 2, 5, 5, 6]
👉 KEEP: [1, 1]

⚡ TURN 5 (Rolls: 1)
🎲 Dice: [1, 1, 1, 3, 4, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 5 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 2]
� SCORE: 1s in Down
   Current Score: 271

⚡ TURN 6 (Rolls: 2)
🎲 Dice: [5, 6, 6, 6, 6, 6]
� SCORE: Y in Anno
   Current Score: 361

⚡ TURN 7 (Rolls: 2)
🎲 Dice: [1, 3, 3, 3, 5, 6]
👉 KEEP: [6]

⚡ TURN 7 (Rolls: 1)
🎲 Dice: [2, 2, 4, 5, 6, 6]
👉 KEEP: [2, 2, 6, 6]

⚡ TURN 7 (Rolls: 0)
🎲 Dice: [2, 2, 2, 3, 6, 6]
� SCORE: 2s in Down
   Current Score: 367

⚡ TURN 8 (Rolls: 2)
🎲 Dice: [1, 2, 2, 4, 4, 6]
👉 KEEP: [6]

⚡ TURN 8 (Rolls: 1)
🎲 Dice: [2, 2, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 8 (Rolls: 0)
🎲 Dice: [1, 6, 6, 6, 6, 6]
� SCORE: Y in Free
   Current Score: 457

⚡ TURN 9 (Rolls: 2)
🎲 Dice: [1, 2, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 9 (Rolls: 1)
🎲 Dice: [1, 2, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 9 (Rolls: 0)
🎲 Dice: [1, 3, 5, 6, 6, 6]
� SCORE: T in Free
   Current Score: 495

⚡ TURN 10 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 10 (Rolls: 1)
🎲 Dice: [3, 4, 5, 5, 6, 6]
👉 KEEP: [5, 5, 6, 6]

⚡ TURN 10 (Rolls: 0)
🎲 Dice: [2, 3, 5, 5, 6, 6]
� SCORE: 3s in Down
   Current Score: 498

⚡ TURN 11 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 2, 2]
� SCORE: Min in Anno
   Current Score: 498

⚡ TURN 12 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 5, 6]
👉 KEEP: [4, 6]

⚡ TURN 12 (Rolls: 1)
🎲 Dice: [1, 2, 3, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 12 (Rolls: 0)
🎲 Dice: [2, 4, 4, 6, 6, 6]
� SCORE: F in Up
   Current Score: 564

⚡ TURN 13 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 3, 5]
👉 KEEP: nothing

⚡ TURN 13 (Rolls: 1)
🎲 Dice: [1, 1, 2, 4, 6, 6]
👉 KEEP: nothing

⚡ TURN 13 (Rolls: 0)
🎲 Dice: [3, 4, 5, 5, 6, 6]
� SCORE: Max in Anno
   Current Score: 564

⚡ TURN 14 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 5]
👉 KEEP: [1, 2, 3, 4, 5]

⚡ TURN 14 (Rolls: 1)
🎲 Dice: [1, 2, 3, 4, 4, 5]
👉 KEEP: [1, 2, 3, 4, 5]

⚡ TURN 14 (Rolls: 0)
🎲 Dice: [1, 1, 2, 3, 4, 5]
� SCORE: K in Up
   Current Score: 609

⚡ TURN 15 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 15 (Rolls: 1)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 15 (Rolls: 0)
🎲 Dice: [1, 1, 4, 4, 6, 6]
� SCORE: 4s in Down
   Current Score: 617

⚡ TURN 16 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 16 (Rolls: 1)
🎲 Dice: [1, 2, 4, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 16 (Rolls: 0)
🎲 Dice: [1, 4, 4, 6, 6, 6]
� SCORE: T in Up
   Current Score: 655

⚡ TURN 17 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 3, 4]
👉 KEEP: [1, 1]

⚡ TURN 17 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 4, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 17 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 3]
� SCORE: 1s in Free
   Current Score: 660

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 4]
👉 KEEP: [1, 1, 2]

⚡ TURN 18 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 2, 3]
👉 KEEP: [1, 1, 1, 2, 2]

⚡ TURN 18 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 2, 5]
� SCORE: Min in Up
   Current Score: 660

⚡ TURN 19 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 3, 4]
� ANNOUNCE: 1s

⚡ TURN 19 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 3, 4]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 19 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 1, 5]
� SCORE: 1s in Anno
   Current Score: 760

⚡ TURN 20 (Rolls: 2)
🎲 Dice: [1, 1, 5, 6, 6, 6]
� ANNOUNCE: 6s

⚡ TURN 20 (Rolls: 2)
🎲 Dice: [1, 1, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 20 (Rolls: 1)
🎲 Dice: [2, 3, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 20 (Rolls: 0)
🎲 Dice: [2, 3, 5, 6, 6, 6]
� SCORE: 6s in Anno
   Current Score: 778

⚡ TURN 21 (Rolls: 2)
🎲 Dice: [1, 2, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 21 (Rolls: 1)
🎲 Dice: [2, 2, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 21 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 5, 6]
� SCORE: 5s in Down
   Current Score: 798

⚡ TURN 22 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 22 (Rolls: 1)
🎲 Dice: [2, 2, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 22 (Rolls: 0)
🎲 Dice: [3, 4, 6, 6, 6, 6]
� SCORE: 6s in Down
   Current Score: 852

⚡ TURN 23 (Rolls: 2)
🎲 Dice: [1, 2, 5, 5, 6, 6]
👉 KEEP: [5, 5, 6, 6]

⚡ TURN 23 (Rolls: 1)
🎲 Dice: [3, 4, 5, 5, 6, 6]
👉 KEEP: [4, 5, 5, 6, 6]

⚡ TURN 23 (Rolls: 0)
🎲 Dice: [2, 4, 5, 5, 6, 6]
� SCORE: Max in Down
   Current Score: 852

⚡ TURN 24 (Rolls: 2)
🎲 Dice: [2, 2, 2, 3, 3, 6]
👉 KEEP: nothing

⚡ TURN 24 (Rolls: 1)
🎲 Dice: [1, 3, 4, 6, 6, 6]
� SCORE: T in Anno
   Current Score: 890

⚡ TURN 25 (Rolls: 2)
🎲 Dice: [1, 1, 1, 3, 5, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 25 (Rolls: 1)
🎲 Dice: [1, 1, 1, 4, 5, 5]
👉 KEEP: [1, 1, 1]

⚡ TURN 25 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 6]
� SCORE: Min in Down
   Current Score: 995

⚡ TURN 26 (Rolls: 2)
🎲 Dice: [1, 2, 2, 4, 5, 6]
👉 KEEP: [5, 6]

⚡ TURN 26 (Rolls: 1)
🎲 Dice: [5, 5, 6, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6, 6]

⚡ TURN 26 (Rolls: 0)
🎲 Dice: [4, 5, 6, 6, 6, 6]
� SCORE: Max in Up
   Current Score: 995

⚡ TURN 27 (Rolls: 2)
🎲 Dice: [2, 3, 3, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 27 (Rolls: 1)
🎲 Dice: [1, 1, 2, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 27 (Rolls: 0)
🎲 Dice: [5, 5, 5, 5, 6, 6]
� SCORE: T in Down
   Current Score: 1030

⚡ TURN 28 (Rolls: 2)
🎲 Dice: [1, 3, 3, 5, 5, 6]
👉 KEEP: [6]

⚡ TURN 28 (Rolls: 1)
🎲 Dice: [2, 4, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 28 (Rolls: 0)
🎲 Dice: [2, 3, 6, 6, 6, 6]
� SCORE: 6s in Up
   Current Score: 1054

⚡ TURN 29 (Rolls: 2)
🎲 Dice: [1, 1, 1, 3, 4, 4]
👉 KEEP: [1, 1, 1]

⚡ TURN 29 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 3, 4]
👉 KEEP: [1, 1, 1, 2, 3]

⚡ TURN 29 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 3, 4]
� SCORE: Min in Free
   Current Score: 1054

⚡ TURN 30 (Rolls: 2)
🎲 Dice: [2, 2, 3, 3, 4, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 30 (Rolls: 1)
🎲 Dice: [2, 3, 3, 4, 5, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 30 (Rolls: 0)
🎲 Dice: [2, 2, 3, 4, 5, 5]
� SCORE: 2s in Free
   Current Score: 1058

⚡ TURN 31 (Rolls: 2)
🎲 Dice: [1, 2, 4, 4, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 31 (Rolls: 1)
🎲 Dice: [2, 2, 3, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 31 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 5, 5]
� SCORE: 5s in Up
   Current Score: 1083

⚡ TURN 32 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 6]
👉 KEEP: [4, 4]

⚡ TURN 32 (Rolls: 1)
🎲 Dice: [4, 4, 4, 5, 5, 6]
👉 KEEP: [4, 4, 4]

⚡ TURN 32 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 6, 6]
� SCORE: 4s in Up
   Current Score: 1125

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [2, 3, 3, 4, 5, 6]
� SCORE: K in Down
   Current Score: 1175

⚡ TURN 34 (Rolls: 2)
🎲 Dice: [1, 1, 1, 4, 4, 5]
👉 KEEP: [4, 4]

⚡ TURN 34 (Rolls: 1)
🎲 Dice: [1, 1, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4, 5]

⚡ TURN 34 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 4, 5]
� SCORE: 4s in Free
   Current Score: 1191

⚡ TURN 35 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 4, 6]
👉 KEEP: [3, 6]

⚡ TURN 35 (Rolls: 1)
🎲 Dice: [1, 3, 3, 4, 5, 6]
👉 KEEP: [3, 3]

⚡ TURN 35 (Rolls: 0)
🎲 Dice: [2, 2, 3, 3, 5, 6]
� SCORE: 3s in Up
   Current Score: 1197

⚡ TURN 36 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 36 (Rolls: 1)
🎲 Dice: [1, 2, 3, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 36 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 6, 6]
� SCORE: F in Down
   Current Score: 1264

⚡ TURN 37 (Rolls: 2)
🎲 Dice: [1, 2, 2, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 37 (Rolls: 1)
🎲 Dice: [2, 4, 6, 6, 6, 6]
� SCORE: P in Down
   Current Score: 1338

⚡ TURN 38 (Rolls: 2)
🎲 Dice: [2, 2, 3, 4, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 38 (Rolls: 1)
🎲 Dice: [2, 3, 4, 5, 5, 6]
� SCORE: K in Free
   Current Score: 1388

⚡ TURN 39 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 5]
👉 KEEP: nothing

⚡ TURN 39 (Rolls: 1)
🎲 Dice: [3, 5, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 39 (Rolls: 0)
🎲 Dice: [2, 3, 5, 5, 5, 5]
� SCORE: P in Free
   Current Score: 1458

⚡ TURN 40 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 40 (Rolls: 1)
🎲 Dice: [4, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 40 (Rolls: 0)
🎲 Dice: [1, 2, 5, 5, 5, 5]
� SCORE: 5s in Free
   Current Score: 1478

⚡ TURN 41 (Rolls: 2)
🎲 Dice: [3, 4, 4, 4, 6, 6]
� ANNOUNCE: F

⚡ TURN 41 (Rolls: 2)
🎲 Dice: [3, 4, 4, 4, 6, 6]
� SCORE: F in Anno
   Current Score: 1542

⚡ TURN 42 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 42 (Rolls: 1)
🎲 Dice: [2, 3, 3, 4, 4, 6]
👉 KEEP: nothing

⚡ TURN 42 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 4, 6]
� SCORE: 4s in Anno
   Current Score: 1558

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 43 (Rolls: 1)
🎲 Dice: [4, 4, 5, 5, 6, 6]
👉 KEEP: [5, 5, 6, 6]

⚡ TURN 43 (Rolls: 0)
🎲 Dice: [2, 5, 5, 6, 6, 6]
� SCORE: F in Free
   Current Score: 1626

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 4, 5]
� ANNOUNCE: 3s

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 4, 5]
👉 KEEP: [3, 3]

⚡ TURN 44 (Rolls: 1)
🎲 Dice: [2, 3, 3, 3, 4, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 44 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 3, 5]
� SCORE: 3s in Anno
   Current Score: 1638

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [1, 2, 5, 5, 5, 6]
� ANNOUNCE: 5s

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [1, 2, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 45 (Rolls: 1)
🎲 Dice: [2, 2, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 45 (Rolls: 0)
🎲 Dice: [3, 5, 5, 5, 5, 6]
� SCORE: 5s in Anno
   Current Score: 1688

⚡ TURN 46 (Rolls: 2)
🎲 Dice: [3, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 46 (Rolls: 1)
🎲 Dice: [1, 3, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 46 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 5, 5]
� SCORE: Y in Down
   Current Score: 1773

⚡ TURN 47 (Rolls: 2)
🎲 Dice: [2, 2, 2, 3, 4, 5]
👉 KEEP: [2, 2, 2]

⚡ TURN 47 (Rolls: 1)
🎲 Dice: [2, 2, 2, 4, 4, 6]
👉 KEEP: [2, 2, 2]

⚡ TURN 47 (Rolls: 0)
🎲 Dice: [2, 2, 2, 3, 4, 6]
� SCORE: 2s in Up
   Current Score: 1779

⚡ TURN 48 (Rolls: 2)
🎲 Dice: [2, 2, 4, 4, 5, 5]
👉 KEEP: nothing

⚡ TURN 48 (Rolls: 1)
🎲 Dice: [2, 2, 3, 3, 4, 5]
� SCORE: 2s in Anno
   Current Score: 1783

⚡ TURN 49 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 49 (Rolls: 1)
🎲 Dice: [2, 3, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 49 (Rolls: 0)
🎲 Dice: [3, 6, 6, 6, 6, 6]
� SCORE: 6s in Free
   Current Score: 1843

⚡ TURN 50 (Rolls: 2)
🎲 Dice: [1, 2, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 50 (Rolls: 1)
🎲 Dice: [4, 5, 5, 6, 6, 6]
👉 KEEP: [5, 5, 6, 6, 6]

⚡ TURN 50 (Rolls: 0)
🎲 Dice: [5, 5, 5, 6, 6, 6]
� SCORE: Max in Free
   Current Score: 1943

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 6, 6]
👉 KEEP: [3]

⚡ TURN 51 (Rolls: 1)
🎲 Dice: [1, 3, 3, 6, 6, 6]
👉 KEEP: [3, 3]

⚡ TURN 51 (Rolls: 0)
🎲 Dice: [3, 3, 3, 5, 5, 6]
� SCORE: 3s in Free
   Current Score: 1952

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [1, 1, 2, 2, 4, 6]
👉 KEEP: [1, 1]

⚡ TURN 52 (Rolls: 3)
🎲 Dice: [1, 1, 1, 3, 3, 5]
👉 KEEP: [1, 1, 1]

⚡ TURN 52 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 3, 4]
👉 KEEP: [1, 1, 1]

⚡ TURN 52 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 5, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 52 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 3, 3]
� SCORE: 1s in Up
   Current Score: 2044

        Down     Free      Up      Anno  
      ------------------------------------
  1s |    5        5        4        5     
  2s |    6        4        6        4     
  3s |    3        9        6        12    
  4s |    8        16       12       16    
  5s |    20       20       25       20    
  6s |    24       30       24       18    
 Max |    26       28       29       26    
 Min |    5        8        7        7     
   T |    35       38       38       38    
   K |    50       50       45       50    
   F |    67       68       66       64    
   P |    74       70       66       70    
   Y |    85       90       80       90    
      ------------------------------------
🏁 FINAL SCORE: 2044

```
