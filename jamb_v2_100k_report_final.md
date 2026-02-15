# Jamb Agent V2 Evaluation Report
**Games:** 100,000 | **Model:** `ckpt_1469054976.npz`  
**Device:** GPU (via WSL2 JAX)

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **1704.47** |
| **Max** | **2018** |
| Median | 1713.0 |
| StdDev | 108.83 |
| Min | 1150 |

### Percentiles
| % | Score |
|---|---|
| 1% | 1418 |
| 10% | 1560 |
| 25% | 1636 |
| 50% | 1713 |
| 75% | 1782 |
| 90% | 1839 |
| 99% | 1920 |

## ⏱️ Column Completion Speed
Average turn number when the column was fully filled (Lower is faster, but usually constrained by rules).
For 'Up' column, it fills bottom-to-top, so 'faster' means finishing 1s earlier.
Wait, game ends around turn 50-60.

| Column | Avg Turn Filled |
|:---|:---|
| **Down** | 45.5 |
| **Free** | 48.4 |
| **Up** | 50.3 |
| **Anno** | 50.4 |

## 🎲 Average Board Values
(Averaged across 100k games)

| Row | Down | Free | Up | Anno |
|:----|:---:|:---:|:---:|:---:|
| **1s** | 3.48 | 4.01 | 3.35 | 3.87 |
| **2s** | 5.16 | 4.42 | 4.78 | 3.42 |
| **3s** | 8.67 | 8.66 | 8.03 | 8.41 |
| **4s** | 12.06 | 13.03 | 11.44 | 12.66 |
| **5s** | 15.28 | 16.52 | 14.74 | 16.72 |
| **6s** | 19.42 | 20.68 | 18.68 | 20.38 |
| **Max** | 26.14 | 26.26 | 25.78 | 26.03 |
| **Min** | 8.69 | 8.30 | 8.71 | 8.21 |
| **T** | 34.55 | 35.21 | 34.48 | 32.68 |
| **K** | 48.35 | 48.57 | 47.99 | 48.55 |
| **F** | 63.23 | 63.54 | 63.68 | 61.06 |
| **P** | 67.96 | 71.09 | 68.98 | 61.69 |
| **Y** | 57.66 | 79.15 | 72.52 | 39.49 |

## 📜 Best Game Log (Score: 2018)
Seed: `579250241`

```text
--- Replaying Game with Seed 579250241 ---

⚡ TURN 1 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 3, 4]
👉 KEEP: [1, 1, 1]

⚡ TURN 1 (Rolls: 1)
🎲 Dice: [1, 1, 1, 3, 4, 5]
👉 KEEP: [1, 1, 1]

⚡ TURN 1 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 2]
� SCORE: 1s in Down
   Current Score: 5

⚡ TURN 2 (Rolls: 2)
🎲 Dice: [2, 2, 3, 4, 5, 5]
👉 KEEP: [2, 2]

⚡ TURN 2 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 5, 6]
👉 KEEP: [2, 2]

⚡ TURN 2 (Rolls: 0)
🎲 Dice: [1, 2, 2, 2, 5, 6]
� SCORE: 2s in Down
   Current Score: 11

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 3, 6]
� ANNOUNCE: 1s

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [1, 1, 1, 1, 3, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 3 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 1, 2]
👉 KEEP: [1, 1, 1, 1, 1]

⚡ TURN 3 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 3]
� SCORE: 1s in Anno
   Current Score: 16

⚡ TURN 4 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 6]
� SCORE: K in Anno
   Current Score: 66

⚡ TURN 5 (Rolls: 2)
🎲 Dice: [3, 3, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 5 (Rolls: 1)
🎲 Dice: [1, 2, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 5 (Rolls: 0)
🎲 Dice: [5, 5, 5, 5, 5, 6]
� SCORE: Y in Up
   Current Score: 151

⚡ TURN 6 (Rolls: 2)
🎲 Dice: [1, 1, 2, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 6 (Rolls: 1)
🎲 Dice: [2, 3, 3, 4, 6, 6]
👉 KEEP: [3, 3]

⚡ TURN 6 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 4, 5]
� SCORE: 3s in Down
   Current Score: 160

⚡ TURN 7 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 4, 6]
👉 KEEP: [4]

⚡ TURN 7 (Rolls: 1)
🎲 Dice: [1, 2, 3, 3, 3, 4]
👉 KEEP: [3, 3, 3]

⚡ TURN 7 (Rolls: 0)
🎲 Dice: [2, 3, 3, 3, 3, 6]
� SCORE: P in Up
   Current Score: 222

⚡ TURN 8 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 2, 3]
👉 KEEP: [1, 1, 1]

⚡ TURN 8 (Rolls: 1)
🎲 Dice: [1, 1, 1, 5, 5, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 8 (Rolls: 0)
🎲 Dice: [1, 1, 1, 4, 4, 5]
� SCORE: 4s in Down
   Current Score: 230

⚡ TURN 9 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 9 (Rolls: 1)
🎲 Dice: [3, 3, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 9 (Rolls: 0)
🎲 Dice: [2, 6, 6, 6, 6, 6]
� SCORE: Y in Free
   Current Score: 320

⚡ TURN 10 (Rolls: 2)
🎲 Dice: [1, 2, 4, 4, 5, 6]
👉 KEEP: [5, 6]

⚡ TURN 10 (Rolls: 1)
🎲 Dice: [2, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 10 (Rolls: 0)
🎲 Dice: [3, 3, 5, 5, 5, 6]
� SCORE: F in Up
   Current Score: 381

⚡ TURN 11 (Rolls: 2)
🎲 Dice: [1, 3, 3, 3, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 11 (Rolls: 1)
🎲 Dice: [1, 2, 4, 4, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 11 (Rolls: 0)
🎲 Dice: [2, 4, 4, 5, 5, 5]
� SCORE: 5s in Down
   Current Score: 396

⚡ TURN 12 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 5, 6]
👉 KEEP: [2, 3, 5, 6]

⚡ TURN 12 (Rolls: 1)
🎲 Dice: [2, 3, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 12 (Rolls: 0)
🎲 Dice: [1, 2, 3, 6, 6, 6]
� SCORE: 6s in Down
   Current Score: 444

⚡ TURN 13 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 6, 6]
� SCORE: K in Up
   Current Score: 494

⚡ TURN 14 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 4, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 14 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 3, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 14 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 3, 6]
� SCORE: 1s in Free
   Current Score: 498

⚡ TURN 15 (Rolls: 2)
🎲 Dice: [1, 2, 4, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 15 (Rolls: 1)
🎲 Dice: [3, 5, 5, 6, 6, 6]
👉 KEEP: [5, 5, 6, 6, 6]

⚡ TURN 15 (Rolls: 0)
🎲 Dice: [2, 5, 5, 6, 6, 6]
� SCORE: Max in Down
   Current Score: 498

⚡ TURN 16 (Rolls: 2)
🎲 Dice: [2, 3, 3, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 16 (Rolls: 1)
🎲 Dice: [3, 4, 5, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 16 (Rolls: 0)
🎲 Dice: [1, 3, 5, 5, 6, 6]
� SCORE: Max in Free
   Current Score: 498

⚡ TURN 17 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 4, 6]
👉 KEEP: [1, 1]

⚡ TURN 17 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 4, 5]
👉 KEEP: [1, 1, 1, 2]

⚡ TURN 17 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 4, 5]
� SCORE: Min in Down
   Current Score: 593

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 3, 3, 3, 6, 6]
� ANNOUNCE: F

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 3, 3, 3, 6, 6]
� SCORE: F in Anno
   Current Score: 654

⚡ TURN 19 (Rolls: 2)
🎲 Dice: [1, 2, 2, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 19 (Rolls: 1)
🎲 Dice: [4, 4, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 19 (Rolls: 0)
🎲 Dice: [1, 2, 6, 6, 6, 6]
� SCORE: T in Down
   Current Score: 692

⚡ TURN 20 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 20 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 20 (Rolls: 0)
🎲 Dice: [2, 3, 4, 5, 5, 6]
� SCORE: K in Down
   Current Score: 742

⚡ TURN 21 (Rolls: 2)
🎲 Dice: [1, 2, 2, 5, 5, 6]
👉 KEEP: [5, 5]

⚡ TURN 21 (Rolls: 1)
🎲 Dice: [1, 1, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 21 (Rolls: 0)
🎲 Dice: [2, 3, 5, 5, 5, 5]
� SCORE: P in Free
   Current Score: 812

⚡ TURN 22 (Rolls: 2)
🎲 Dice: [4, 5, 5, 5, 5, 6]
� ANNOUNCE: P

⚡ TURN 22 (Rolls: 2)
🎲 Dice: [4, 5, 5, 5, 5, 6]
� SCORE: P in Anno
   Current Score: 882

⚡ TURN 23 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 5]
👉 KEEP: [4, 4, 5]

⚡ TURN 23 (Rolls: 1)
🎲 Dice: [4, 4, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4, 4, 4]

⚡ TURN 23 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 4, 4]
� SCORE: 4s in Free
   Current Score: 902

⚡ TURN 24 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 5, 6]
👉 KEEP: [5, 5]

⚡ TURN 24 (Rolls: 1)
🎲 Dice: [1, 2, 2, 4, 5, 5]
👉 KEEP: [5, 5]

⚡ TURN 24 (Rolls: 0)
🎲 Dice: [1, 1, 2, 5, 5, 6]
� SCORE: 2s in Free
   Current Score: 904

⚡ TURN 25 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 6]
👉 KEEP: [1, 5, 6]

⚡ TURN 25 (Rolls: 1)
🎲 Dice: [1, 5, 5, 6, 6, 6]
👉 KEEP: [5, 5, 6, 6, 6]

⚡ TURN 25 (Rolls: 0)
🎲 Dice: [2, 5, 5, 6, 6, 6]
� SCORE: F in Down
   Current Score: 972

⚡ TURN 26 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 5, 6]
👉 KEEP: [6]

⚡ TURN 26 (Rolls: 1)
🎲 Dice: [3, 3, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 26 (Rolls: 0)
🎲 Dice: [1, 3, 6, 6, 6, 6]
� SCORE: P in Down
   Current Score: 1046

⚡ TURN 27 (Rolls: 2)
🎲 Dice: [1, 1, 3, 3, 4, 5]
👉 KEEP: [3, 5]

⚡ TURN 27 (Rolls: 1)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 27 (Rolls: 0)
🎲 Dice: [3, 4, 4, 6, 6, 6]
� SCORE: T in Up
   Current Score: 1084

⚡ TURN 28 (Rolls: 2)
🎲 Dice: [4, 5, 6, 6, 6, 6]
� ANNOUNCE: Y

⚡ TURN 28 (Rolls: 2)
🎲 Dice: [4, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 28 (Rolls: 1)
🎲 Dice: [5, 5, 6, 6, 6, 6]
👉 KEEP: [6, 6, 6, 6]

⚡ TURN 28 (Rolls: 0)
🎲 Dice: [1, 6, 6, 6, 6, 6]
� SCORE: Y in Anno
   Current Score: 1174

⚡ TURN 29 (Rolls: 2)
🎲 Dice: [2, 2, 3, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 29 (Rolls: 1)
🎲 Dice: [4, 5, 5, 5, 6, 6]
👉 KEEP: [5, 5, 5, 6, 6]

⚡ TURN 29 (Rolls: 0)
🎲 Dice: [1, 5, 5, 5, 6, 6]
� SCORE: F in Free
   Current Score: 1241

⚡ TURN 30 (Rolls: 2)
🎲 Dice: [2, 2, 3, 3, 3, 6]
� ANNOUNCE: 3s

⚡ TURN 30 (Rolls: 2)
🎲 Dice: [2, 2, 3, 3, 3, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 30 (Rolls: 1)
🎲 Dice: [1, 2, 3, 3, 3, 5]
👉 KEEP: [3, 3, 3]

⚡ TURN 30 (Rolls: 0)
🎲 Dice: [1, 2, 3, 3, 3, 3]
� SCORE: 3s in Anno
   Current Score: 1253

⚡ TURN 31 (Rolls: 2)
🎲 Dice: [1, 3, 3, 4, 4, 6]
👉 KEEP: [1]

⚡ TURN 31 (Rolls: 1)
🎲 Dice: [1, 1, 2, 3, 5, 6]
👉 KEEP: [1, 1, 2, 3]

⚡ TURN 31 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 3, 5]
� SCORE: Min in Up
   Current Score: 1253

⚡ TURN 32 (Rolls: 2)
🎲 Dice: [2, 3, 3, 3, 5, 5]
👉 KEEP: [3, 3, 3]

⚡ TURN 32 (Rolls: 1)
🎲 Dice: [3, 3, 3, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 32 (Rolls: 0)
🎲 Dice: [1, 3, 4, 5, 6, 6]
� SCORE: Max in Up
   Current Score: 1253

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 5, 5]
� ANNOUNCE: 5s

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 33 (Rolls: 1)
🎲 Dice: [1, 2, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 33 (Rolls: 0)
🎲 Dice: [2, 2, 4, 5, 5, 5]
� SCORE: 5s in Anno
   Current Score: 1268

⚡ TURN 34 (Rolls: 2)
🎲 Dice: [1, 4, 4, 4, 4, 5]
� ANNOUNCE: 4s

⚡ TURN 34 (Rolls: 2)
🎲 Dice: [1, 4, 4, 4, 4, 5]
👉 KEEP: [1, 4, 4, 4, 4]

⚡ TURN 34 (Rolls: 1)
🎲 Dice: [1, 2, 4, 4, 4, 4]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 34 (Rolls: 0)
🎲 Dice: [1, 4, 4, 4, 4, 6]
� SCORE: 4s in Anno
   Current Score: 1284

⚡ TURN 35 (Rolls: 2)
🎲 Dice: [1, 3, 3, 4, 4, 5]
👉 KEEP: [3, 3]

⚡ TURN 35 (Rolls: 1)
🎲 Dice: [2, 2, 2, 3, 3, 4]
👉 KEEP: nothing

⚡ TURN 35 (Rolls: 0)
🎲 Dice: [1, 3, 4, 5, 5, 5]
� SCORE: T in Anno
   Current Score: 1319

⚡ TURN 36 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 36 (Rolls: 1)
🎲 Dice: [1, 5, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 36 (Rolls: 0)
🎲 Dice: [3, 5, 6, 6, 6, 6]
� SCORE: 6s in Up
   Current Score: 1343

⚡ TURN 37 (Rolls: 2)
🎲 Dice: [1, 2, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 37 (Rolls: 1)
🎲 Dice: [1, 5, 5, 5, 5, 5]
� SCORE: Y in Down
   Current Score: 1428

⚡ TURN 38 (Rolls: 2)
🎲 Dice: [3, 3, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 38 (Rolls: 1)
🎲 Dice: [1, 2, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 38 (Rolls: 0)
🎲 Dice: [5, 5, 5, 5, 6, 6]
� SCORE: 5s in Up
   Current Score: 1448

⚡ TURN 39 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 39 (Rolls: 1)
🎲 Dice: [2, 2, 2, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 39 (Rolls: 0)
🎲 Dice: [2, 3, 4, 6, 6, 6]
� SCORE: T in Free
   Current Score: 1486

⚡ TURN 40 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 5, 6]
👉 KEEP: [1, 1, 4]

⚡ TURN 40 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 4, 5]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 40 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 2, 3]
� SCORE: Min in Free
   Current Score: 1562

⚡ TURN 41 (Rolls: 2)
🎲 Dice: [1, 3, 4, 4, 4, 6]
👉 KEEP: [4, 4, 4]

⚡ TURN 41 (Rolls: 1)
🎲 Dice: [3, 3, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4]

⚡ TURN 41 (Rolls: 0)
🎲 Dice: [1, 2, 4, 4, 4, 4]
� SCORE: 4s in Up
   Current Score: 1608

⚡ TURN 42 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 5, 6]
👉 KEEP: [3, 3]

⚡ TURN 42 (Rolls: 1)
🎲 Dice: [3, 3, 3, 4, 4, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 42 (Rolls: 0)
🎲 Dice: [3, 3, 3, 3, 6, 6]
� SCORE: 3s in Up
   Current Score: 1620

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [4, 5, 5, 5, 6, 6]
� ANNOUNCE: Max

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [4, 5, 5, 5, 6, 6]
👉 KEEP: [5, 5, 5, 6, 6]

⚡ TURN 43 (Rolls: 1)
🎲 Dice: [4, 5, 5, 5, 6, 6]
👉 KEEP: [5, 5, 5, 6, 6]

⚡ TURN 43 (Rolls: 0)
🎲 Dice: [3, 5, 5, 5, 6, 6]
� SCORE: Max in Anno
   Current Score: 1620

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 1, 1, 4, 6, 6]
� ANNOUNCE: Min

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 1, 1, 4, 6, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 44 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 2, 5]
👉 KEEP: [1, 1, 1, 2, 2]

⚡ TURN 44 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 2, 2]
� SCORE: Min in Anno
   Current Score: 1725

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [2, 4, 5, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 45 (Rolls: 1)
🎲 Dice: [2, 4, 6, 6, 6, 6]
👉 KEEP: [4, 6, 6, 6, 6]

⚡ TURN 45 (Rolls: 0)
🎲 Dice: [3, 4, 6, 6, 6, 6]
� SCORE: 6s in Free
   Current Score: 1749

⚡ TURN 46 (Rolls: 2)
🎲 Dice: [2, 3, 3, 3, 4, 6]
👉 KEEP: [2, 3, 4, 6]

⚡ TURN 46 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 6, 6]
👉 KEEP: [2, 2, 3, 4, 6]

⚡ TURN 46 (Rolls: 0)
🎲 Dice: [2, 2, 2, 3, 4, 6]
� SCORE: 2s in Up
   Current Score: 1755

⚡ TURN 47 (Rolls: 2)
🎲 Dice: [1, 1, 2, 4, 5, 6]
👉 KEEP: [1, 1, 5]

⚡ TURN 47 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 2, 5]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 47 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 4]
� SCORE: 1s in Up
   Current Score: 1835

⚡ TURN 48 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 48 (Rolls: 1)
🎲 Dice: [4, 5, 5, 5, 5, 5]
� SCORE: 5s in Free
   Current Score: 1890

⚡ TURN 49 (Rolls: 2)
🎲 Dice: [2, 3, 4, 4, 5, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 49 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 49 (Rolls: 0)
🎲 Dice: [2, 2, 3, 4, 5, 6]
� SCORE: K in Free
   Current Score: 1940

⚡ TURN 50 (Rolls: 2)
🎲 Dice: [1, 1, 4, 4, 5, 5]
👉 KEEP: nothing

⚡ TURN 50 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 2, 6]
� SCORE: 2s in Anno
   Current Score: 1946

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 5, 6]
👉 KEEP: [3, 3]

⚡ TURN 51 (Rolls: 1)
🎲 Dice: [3, 3, 3, 4, 6, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 51 (Rolls: 0)
🎲 Dice: [1, 3, 3, 3, 3, 5]
� SCORE: 3s in Free
   Current Score: 1958

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [3, 3, 4, 4, 5, 6]
� ANNOUNCE: 6s

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [3, 3, 4, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 52 (Rolls: 3)
🎲 Dice: [1, 2, 3, 6, 6, 6]
👉 KEEP: [1, 6, 6, 6]

⚡ TURN 52 (Rolls: 2)
🎲 Dice: [1, 4, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 52 (Rolls: 1)
🎲 Dice: [2, 2, 4, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 52 (Rolls: 0)
🎲 Dice: [4, 6, 6, 6, 6, 6]
� SCORE: 6s in Anno
   Current Score: 2018

        Down     Free      Up      Anno  
      ------------------------------------
  1s |    5        4        5        5     
  2s |    6        2        6        6     
  3s |    9        12       12       12    
  4s |    8        20       16       16    
  5s |    15       25       20       15    
  6s |    18       24       24       30    
 Max |    28       25       24       27    
 Min |    9        6        9        6     
   T |    38       38       38       35    
   K |    50       50       50       50    
   F |    68       67       61       61    
   P |    74       70       62       70    
   Y |    85       90       85       90    
      ------------------------------------
🏁 FINAL SCORE: 2018

```
