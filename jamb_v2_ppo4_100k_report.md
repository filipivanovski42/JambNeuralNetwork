# Jamb Agent (V2 PPO_4) Evaluation Report
**Games:** 100,000 | **Model:** `ckpt_1940914176.npz`  
**Device:** GPU (via WSL2 JAX)

## 🏆 Score Statistics

| Metric | Value |
|:---|:---|
| **Average** | **1638.43** |
| **Max** | **1991** |
| Median | 1645.0 |
| StdDev | 116.08 |
| Min | 1097 |

### Percentiles
| % | Score |
|---|---|
| 1% | 1344 |
| 10% | 1485 |
| 25% | 1563 |
| 50% | 1645 |
| 75% | 1722 |
| 90% | 1783 |
| 99% | 1875 |

## ⏱️ Column Completion Speed
| Column | Avg Turn Filled |
|:---|:---|
| **Down** | 45.5 |
| **Free** | 44.0 |
| **Up** | 49.7 |
| **Anno** | 51.2 |

## 🎲 Average Board Values
| Row | Down | Free | Up | Anno |
|:----|:---:|:---:|:---:|:---:|
| **1s** | 3.29 | 3.91 | 3.32 | 3.80 |
| **2s** | 4.83 | 4.37 | 4.59 | 3.56 |
| **3s** | 8.09 | 8.56 | 7.45 | 8.21 |
| **4s** | 11.36 | 12.22 | 10.60 | 12.40 |
| **5s** | 14.85 | 16.55 | 14.09 | 16.13 |
| **6s** | 18.51 | 20.36 | 17.85 | 19.83 |
| **Max** | 25.83 | 26.21 | 25.36 | 26.02 |
| **Min** | 9.15 | 8.46 | 9.08 | 8.46 |
| **T** | 34.36 | 35.66 | 34.36 | 31.95 |
| **K** | 48.07 | 48.41 | 47.57 | 48.37 |
| **F** | 62.75 | 63.44 | 63.00 | 60.48 |
| **P** | 67.23 | 70.31 | 68.49 | 55.29 |
| **Y** | 54.04 | 76.68 | 62.74 | 35.14 |

## 📜 Best Game Log (Score: 1991)
Seed: `2103316`

```text
--- Replaying Game with Seed 2103316 ---

⚡ TURN 1 (Rolls: 2)
🎲 Dice: [1, 4, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 1 (Rolls: 1)
🎲 Dice: [1, 3, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 1 (Rolls: 0)
🎲 Dice: [1, 3, 5, 5, 5, 5]
📝 SCORE: P in Free
   Current Score: 70

⚡ TURN 2 (Rolls: 2)
🎲 Dice: [1, 3, 3, 4, 4, 5]
👉 KEEP: [1]

⚡ TURN 2 (Rolls: 1)
🎲 Dice: [1, 2, 3, 4, 4, 4]
👉 KEEP: [4, 4, 4]

⚡ TURN 2 (Rolls: 0)
🎲 Dice: [2, 4, 4, 4, 6, 6]
📝 SCORE: F in Free
   Current Score: 134

⚡ TURN 3 (Rolls: 2)
🎲 Dice: [2, 3, 3, 4, 6, 6]
👉 KEEP: [3, 6, 6]

⚡ TURN 3 (Rolls: 1)
🎲 Dice: [3, 6, 6, 6, 6, 6]
📝 SCORE: Y in Up
   Current Score: 224

⚡ TURN 4 (Rolls: 2)
🎲 Dice: [1, 1, 2, 5, 5, 6]
👉 KEEP: [1, 1]

⚡ TURN 4 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 3, 5]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 4 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 2]
📝 SCORE: 1s in Down
   Current Score: 229

⚡ TURN 5 (Rolls: 2)
🎲 Dice: [2, 2, 3, 5, 6, 6]
👉 KEEP: [2, 2, 3, 6, 6]

⚡ TURN 5 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 6, 6]
👉 KEEP: [2, 2]

⚡ TURN 5 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 5, 5]
📝 SCORE: 2s in Down
   Current Score: 233

⚡ TURN 6 (Rolls: 2)
🎲 Dice: [2, 3, 3, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 6 (Rolls: 1)
🎲 Dice: [4, 4, 5, 5, 5, 5]
👉 KEEP: [5, 5, 5, 5]

⚡ TURN 6 (Rolls: 0)
🎲 Dice: [5, 5, 5, 5, 5, 6]
📝 SCORE: Y in Free
   Current Score: 318

⚡ TURN 7 (Rolls: 2)
🎲 Dice: [3, 3, 3, 4, 5, 6]
👉 KEEP: [3, 3, 3, 4]

⚡ TURN 7 (Rolls: 1)
🎲 Dice: [3, 3, 3, 4, 4, 4]
👉 KEEP: [3, 3, 3]

⚡ TURN 7 (Rolls: 0)
🎲 Dice: [3, 3, 3, 3, 4, 6]
📝 SCORE: 3s in Down
   Current Score: 330

⚡ TURN 8 (Rolls: 2)
🎲 Dice: [1, 2, 2, 3, 3, 5]
👉 KEEP: [5]

⚡ TURN 8 (Rolls: 1)
🎲 Dice: [1, 2, 2, 4, 5, 5]
👉 KEEP: [2, 4]

⚡ TURN 8 (Rolls: 0)
🎲 Dice: [1, 2, 2, 3, 4, 5]
📝 SCORE: K in Free
   Current Score: 375

⚡ TURN 9 (Rolls: 2)
🎲 Dice: [3, 3, 4, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 9 (Rolls: 1)
🎲 Dice: [4, 5, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 9 (Rolls: 0)
🎲 Dice: [1, 5, 5, 6, 6, 6]
📝 SCORE: Max in Free
   Current Score: 375

⚡ TURN 10 (Rolls: 2)
🎲 Dice: [2, 3, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 10 (Rolls: 1)
🎲 Dice: [2, 4, 6, 6, 6, 6]
👉 KEEP: [4, 6, 6, 6, 6]

⚡ TURN 10 (Rolls: 0)
🎲 Dice: [4, 5, 6, 6, 6, 6]
📝 SCORE: P in Up
   Current Score: 449

⚡ TURN 11 (Rolls: 2)
🎲 Dice: [2, 5, 5, 5, 5, 5]
📢 ANNOUNCE: Y

⚡ TURN 11 (Rolls: 2)
🎲 Dice: [2, 5, 5, 5, 5, 5]
📝 SCORE: Y in Anno
   Current Score: 534

⚡ TURN 12 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 6, 6]
👉 KEEP: [5, 6, 6]

⚡ TURN 12 (Rolls: 1)
🎲 Dice: [1, 1, 5, 6, 6, 6]
👉 KEEP: [1, 1, 6, 6, 6]

⚡ TURN 12 (Rolls: 0)
🎲 Dice: [1, 1, 5, 6, 6, 6]
📝 SCORE: F in Up
   Current Score: 594

⚡ TURN 13 (Rolls: 2)
🎲 Dice: [2, 3, 4, 4, 5, 5]
👉 KEEP: [4, 4]

⚡ TURN 13 (Rolls: 1)
🎲 Dice: [3, 4, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 13 (Rolls: 0)
🎲 Dice: [1, 2, 4, 4, 4, 4]
📝 SCORE: 4s in Down
   Current Score: 610

⚡ TURN 14 (Rolls: 2)
🎲 Dice: [1, 2, 3, 3, 4, 6]
👉 KEEP: [1]

⚡ TURN 14 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 5, 6]
👉 KEEP: [1, 1, 2, 2]

⚡ TURN 14 (Rolls: 0)
🎲 Dice: [1, 1, 2, 2, 4, 5]
📝 SCORE: Min in Free
   Current Score: 610

⚡ TURN 15 (Rolls: 2)
🎲 Dice: [1, 1, 1, 4, 5, 6]
📢 ANNOUNCE: 1s

⚡ TURN 15 (Rolls: 2)
🎲 Dice: [1, 1, 1, 4, 5, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 15 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 5, 6]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 15 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 1, 4]
📝 SCORE: 1s in Anno
   Current Score: 615

⚡ TURN 16 (Rolls: 2)
🎲 Dice: [1, 2, 2, 4, 4, 6]
👉 KEEP: [1]

⚡ TURN 16 (Rolls: 1)
🎲 Dice: [1, 1, 4, 5, 6, 6]
👉 KEEP: nothing

⚡ TURN 16 (Rolls: 0)
🎲 Dice: [2, 2, 3, 4, 4, 6]
📝 SCORE: 2s in Anno
   Current Score: 619

⚡ TURN 17 (Rolls: 2)
🎲 Dice: [2, 3, 4, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 17 (Rolls: 1)
🎲 Dice: [2, 5, 6, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6, 6]

⚡ TURN 17 (Rolls: 0)
🎲 Dice: [5, 6, 6, 6, 6, 6]
📝 SCORE: 6s in Free
   Current Score: 649

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 4, 4, 4, 4, 5]
📢 ANNOUNCE: P

⚡ TURN 18 (Rolls: 2)
🎲 Dice: [1, 4, 4, 4, 4, 5]
👉 KEEP: [1, 4, 4, 4, 4]

⚡ TURN 18 (Rolls: 1)
🎲 Dice: [1, 1, 4, 4, 4, 4]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 18 (Rolls: 0)
🎲 Dice: [4, 4, 4, 4, 5, 6]
📝 SCORE: P in Anno
   Current Score: 715

⚡ TURN 19 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 5]
👉 KEEP: [2, 3, 4, 5]

⚡ TURN 19 (Rolls: 1)
🎲 Dice: [2, 3, 4, 4, 5, 6]
👉 KEEP: [2, 3, 4, 5, 6]

⚡ TURN 19 (Rolls: 0)
🎲 Dice: [1, 2, 3, 4, 5, 6]
📝 SCORE: K in Up
   Current Score: 765

⚡ TURN 20 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 3, 6]
👉 KEEP: [1, 1]

⚡ TURN 20 (Rolls: 1)
🎲 Dice: [1, 1, 2, 3, 5, 6]
👉 KEEP: [2, 5]

⚡ TURN 20 (Rolls: 0)
🎲 Dice: [2, 2, 3, 5, 6, 6]
📝 SCORE: 5s in Down
   Current Score: 770

⚡ TURN 21 (Rolls: 2)
🎲 Dice: [2, 3, 3, 3, 5, 5]
📝 SCORE: F in Anno
   Current Score: 829

⚡ TURN 22 (Rolls: 2)
🎲 Dice: [2, 3, 4, 4, 5, 6]
📝 SCORE: K in Anno
   Current Score: 879

⚡ TURN 23 (Rolls: 2)
🎲 Dice: [2, 2, 2, 3, 4, 5]
👉 KEEP: [4]

⚡ TURN 23 (Rolls: 1)
🎲 Dice: [3, 4, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 23 (Rolls: 0)
🎲 Dice: [4, 5, 5, 5, 5, 6]
📝 SCORE: T in Up
   Current Score: 914

⚡ TURN 24 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 24 (Rolls: 1)
🎲 Dice: [1, 2, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 24 (Rolls: 0)
🎲 Dice: [2, 2, 2, 3, 6, 6]
📝 SCORE: 2s in Free
   Current Score: 920

⚡ TURN 25 (Rolls: 2)
🎲 Dice: [1, 2, 2, 2, 2, 3]
👉 KEEP: [1]

⚡ TURN 25 (Rolls: 1)
🎲 Dice: [1, 1, 5, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 25 (Rolls: 0)
🎲 Dice: [1, 3, 4, 6, 6, 6]
📝 SCORE: 6s in Down
   Current Score: 968

⚡ TURN 26 (Rolls: 2)
🎲 Dice: [1, 1, 2, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 26 (Rolls: 1)
🎲 Dice: [4, 6, 6, 6, 6, 6]
📝 SCORE: Max in Down
   Current Score: 968

⚡ TURN 27 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 4, 6]
👉 KEEP: [1, 1, 2]

⚡ TURN 27 (Rolls: 1)
🎲 Dice: [1, 1, 1, 2, 3, 5]
👉 KEEP: [1, 1, 1, 2]

⚡ TURN 27 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 2, 3]
📝 SCORE: Min in Down
   Current Score: 1083

⚡ TURN 28 (Rolls: 2)
🎲 Dice: [2, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 28 (Rolls: 1)
🎲 Dice: [1, 4, 4, 6, 6, 6]
👉 KEEP: [1, 4, 6, 6, 6]

⚡ TURN 28 (Rolls: 0)
🎲 Dice: [1, 3, 4, 6, 6, 6]
📝 SCORE: T in Down
   Current Score: 1121

⚡ TURN 29 (Rolls: 2)
🎲 Dice: [2, 2, 3, 3, 4, 6]
👉 KEEP: [2, 3, 4, 6]

⚡ TURN 29 (Rolls: 1)
🎲 Dice: [2, 2, 3, 4, 6, 6]
👉 KEEP: [2, 3, 4, 6]

⚡ TURN 29 (Rolls: 0)
🎲 Dice: [2, 2, 2, 3, 4, 6]
📝 SCORE: 3s in Free
   Current Score: 1124

⚡ TURN 30 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 6, 6]
👉 KEEP: [1, 2]

⚡ TURN 30 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 3, 5]
👉 KEEP: [1, 1, 2, 2]

⚡ TURN 30 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 2, 4]
📝 SCORE: Min in Up
   Current Score: 1124

⚡ TURN 31 (Rolls: 2)
🎲 Dice: [1, 2, 2, 4, 4, 6]
👉 KEEP: [6]

⚡ TURN 31 (Rolls: 1)
🎲 Dice: [2, 3, 4, 5, 6, 6]
📝 SCORE: K in Down
   Current Score: 1174

⚡ TURN 32 (Rolls: 2)
🎲 Dice: [1, 1, 2, 3, 3, 3]
👉 KEEP: [1, 1]

⚡ TURN 32 (Rolls: 1)
🎲 Dice: [1, 1, 1, 3, 4, 4]
👉 KEEP: [1, 1, 1]

⚡ TURN 32 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 4, 6]
📝 SCORE: 1s in Free
   Current Score: 1250

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [2, 3, 4, 4, 4, 5]
📢 ANNOUNCE: 4s

⚡ TURN 33 (Rolls: 2)
🎲 Dice: [2, 3, 4, 4, 4, 5]
👉 KEEP: [4, 4, 4]

⚡ TURN 33 (Rolls: 1)
🎲 Dice: [1, 2, 4, 4, 4, 6]
👉 KEEP: [4, 4, 4, 6]

⚡ TURN 33 (Rolls: 0)
🎲 Dice: [2, 4, 4, 4, 4, 6]
📝 SCORE: 4s in Anno
   Current Score: 1266

⚡ TURN 34 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 4, 5]
👉 KEEP: [5]

⚡ TURN 34 (Rolls: 1)
🎲 Dice: [1, 3, 4, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 34 (Rolls: 0)
🎲 Dice: [1, 4, 5, 5, 6, 6]
📝 SCORE: Max in Up
   Current Score: 1266

⚡ TURN 35 (Rolls: 2)
🎲 Dice: [1, 1, 3, 4, 4, 6]
👉 KEEP: [4, 4, 6]

⚡ TURN 35 (Rolls: 1)
🎲 Dice: [2, 3, 4, 4, 6, 6]
👉 KEEP: [4, 4, 6, 6]

⚡ TURN 35 (Rolls: 0)
🎲 Dice: [4, 4, 5, 6, 6, 6]
📝 SCORE: F in Down
   Current Score: 1332

⚡ TURN 36 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 4, 5]
📢 ANNOUNCE: Min

⚡ TURN 36 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 4, 5]
👉 KEEP: [1, 1, 2, 2]

⚡ TURN 36 (Rolls: 1)
🎲 Dice: [1, 1, 2, 2, 3, 6]
👉 KEEP: [1, 1, 2, 2, 3]

⚡ TURN 36 (Rolls: 0)
🎲 Dice: [1, 1, 1, 2, 2, 3]
📝 SCORE: Min in Anno
   Current Score: 1332

⚡ TURN 37 (Rolls: 2)
🎲 Dice: [1, 2, 3, 5, 5, 6]
👉 KEEP: [5, 5, 6]

⚡ TURN 37 (Rolls: 1)
🎲 Dice: [2, 2, 3, 5, 5, 6]
👉 KEEP: [5, 5]

⚡ TURN 37 (Rolls: 0)
🎲 Dice: [4, 4, 5, 5, 5, 6]
📝 SCORE: T in Free
   Current Score: 1367

⚡ TURN 38 (Rolls: 2)
🎲 Dice: [1, 2, 3, 4, 5, 5]
👉 KEEP: [4, 5, 5]

⚡ TURN 38 (Rolls: 1)
🎲 Dice: [1, 4, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 38 (Rolls: 0)
🎲 Dice: [1, 5, 5, 5, 5, 6]
📝 SCORE: P in Down
   Current Score: 1437

⚡ TURN 39 (Rolls: 2)
🎲 Dice: [1, 1, 2, 4, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 39 (Rolls: 1)
🎲 Dice: [1, 4, 4, 5, 6, 6]
👉 KEEP: [4, 4, 6, 6]

⚡ TURN 39 (Rolls: 0)
🎲 Dice: [1, 2, 4, 4, 6, 6]
📝 SCORE: 4s in Free
   Current Score: 1445

⚡ TURN 40 (Rolls: 2)
🎲 Dice: [1, 1, 4, 4, 5, 6]
👉 KEEP: [6]

⚡ TURN 40 (Rolls: 1)
🎲 Dice: [1, 1, 3, 3, 5, 6]
👉 KEEP: [6]

⚡ TURN 40 (Rolls: 0)
🎲 Dice: [3, 4, 4, 6, 6, 6]
📝 SCORE: 6s in Up
   Current Score: 1463

⚡ TURN 41 (Rolls: 2)
🎲 Dice: [1, 2, 4, 4, 4, 6]
👉 KEEP: [4, 4, 4]

⚡ TURN 41 (Rolls: 1)
🎲 Dice: [1, 4, 4, 4, 5, 6]
👉 KEEP: [5]

⚡ TURN 41 (Rolls: 0)
🎲 Dice: [2, 4, 5, 5, 5, 6]
📝 SCORE: 5s in Up
   Current Score: 1478

⚡ TURN 42 (Rolls: 2)
🎲 Dice: [1, 1, 4, 4, 4, 4]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 42 (Rolls: 1)
🎲 Dice: [1, 1, 4, 4, 4, 4]
👉 KEEP: [4, 4, 4, 4]

⚡ TURN 42 (Rolls: 0)
🎲 Dice: [4, 4, 4, 4, 4, 6]
📝 SCORE: Y in Down
   Current Score: 1558

⚡ TURN 43 (Rolls: 2)
🎲 Dice: [2, 2, 4, 4, 6, 6]
👉 KEEP: [4, 4]

⚡ TURN 43 (Rolls: 1)
🎲 Dice: [2, 3, 4, 4, 4, 6]
👉 KEEP: [4, 4, 4]

⚡ TURN 43 (Rolls: 0)
🎲 Dice: [2, 3, 4, 4, 4, 5]
📝 SCORE: 4s in Up
   Current Score: 1570

⚡ TURN 44 (Rolls: 2)
🎲 Dice: [1, 1, 2, 2, 3, 6]
👉 KEEP: [3, 6]

⚡ TURN 44 (Rolls: 1)
🎲 Dice: [1, 3, 3, 3, 6, 6]
👉 KEEP: [3, 3, 3]

⚡ TURN 44 (Rolls: 0)
🎲 Dice: [2, 3, 3, 3, 5, 6]
📝 SCORE: 3s in Up
   Current Score: 1579

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
📢 ANNOUNCE: 6s

⚡ TURN 45 (Rolls: 2)
🎲 Dice: [1, 3, 4, 5, 6, 6]
👉 KEEP: [6, 6]

⚡ TURN 45 (Rolls: 1)
🎲 Dice: [2, 3, 3, 6, 6, 6]
👉 KEEP: [3, 6, 6, 6]

⚡ TURN 45 (Rolls: 0)
🎲 Dice: [3, 6, 6, 6, 6, 6]
📝 SCORE: 6s in Anno
   Current Score: 1609

⚡ TURN 46 (Rolls: 2)
🎲 Dice: [3, 4, 5, 5, 5, 6]
👉 KEEP: [5, 5, 5]

⚡ TURN 46 (Rolls: 1)
🎲 Dice: [3, 4, 4, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 46 (Rolls: 0)
🎲 Dice: [1, 5, 5, 5, 5, 6]
📝 SCORE: 5s in Free
   Current Score: 1659

⚡ TURN 47 (Rolls: 2)
🎲 Dice: [1, 2, 4, 4, 5, 6]
👉 KEEP: nothing

⚡ TURN 47 (Rolls: 1)
🎲 Dice: [1, 2, 3, 4, 5, 5]
👉 KEEP: nothing

⚡ TURN 47 (Rolls: 0)
🎲 Dice: [1, 2, 4, 4, 4, 4]
📝 SCORE: T in Anno
   Current Score: 1691

⚡ TURN 48 (Rolls: 2)
🎲 Dice: [2, 2, 3, 3, 4, 6]
👉 KEEP: [2, 2]

⚡ TURN 48 (Rolls: 1)
🎲 Dice: [1, 2, 2, 2, 3, 5]
👉 KEEP: [2, 2, 2]

⚡ TURN 48 (Rolls: 0)
🎲 Dice: [2, 2, 2, 2, 5, 5]
📝 SCORE: 2s in Up
   Current Score: 1729

⚡ TURN 49 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 6, 6]
👉 KEEP: [1, 1, 1]

⚡ TURN 49 (Rolls: 1)
🎲 Dice: [1, 1, 1, 1, 4, 5]
👉 KEEP: [1, 1, 1, 1]

⚡ TURN 49 (Rolls: 0)
🎲 Dice: [1, 1, 1, 1, 4, 5]
📝 SCORE: 1s in Up
   Current Score: 1809

⚡ TURN 50 (Rolls: 2)
🎲 Dice: [1, 2, 3, 6, 6, 6]
📢 ANNOUNCE: Max

⚡ TURN 50 (Rolls: 2)
🎲 Dice: [1, 2, 3, 6, 6, 6]
👉 KEEP: [6, 6, 6]

⚡ TURN 50 (Rolls: 1)
🎲 Dice: [4, 5, 6, 6, 6, 6]
👉 KEEP: [5, 6, 6, 6, 6]

⚡ TURN 50 (Rolls: 0)
🎲 Dice: [5, 6, 6, 6, 6, 6]
📝 SCORE: Max in Anno
   Current Score: 1924

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [2, 3, 3, 5, 5, 6]
📢 ANNOUNCE: 5s

⚡ TURN 51 (Rolls: 2)
🎲 Dice: [2, 3, 3, 5, 5, 6]
👉 KEEP: [5, 5]

⚡ TURN 51 (Rolls: 1)
🎲 Dice: [1, 1, 2, 5, 5, 5]
👉 KEEP: [5, 5, 5]

⚡ TURN 51 (Rolls: 0)
🎲 Dice: [2, 5, 5, 5, 5, 5]
📝 SCORE: 5s in Anno
   Current Score: 1979

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [1, 3, 4, 5, 5, 6]
📢 ANNOUNCE: 3s

⚡ TURN 52 (Rolls: 4)
🎲 Dice: [1, 3, 4, 5, 5, 6]
👉 KEEP: [3]

⚡ TURN 52 (Rolls: 3)
🎲 Dice: [1, 2, 3, 3, 4, 5]
👉 KEEP: [3, 3]

⚡ TURN 52 (Rolls: 2)
🎲 Dice: [1, 1, 1, 2, 3, 3]
👉 KEEP: [3, 3]

⚡ TURN 52 (Rolls: 1)
🎲 Dice: [1, 2, 3, 3, 3, 5]
👉 KEEP: [3, 3, 3, 5]

⚡ TURN 52 (Rolls: 0)
🎲 Dice: [3, 3, 3, 3, 4, 5]
📝 SCORE: 3s in Anno
   Current Score: 1991

        Down     Free      Up      Anno  
      ------------------------------------
  1s |    5        4        4        5     
  2s |    4        6        8        4     
  3s |    12       3        9        12    
  4s |    16       8        12       16    
  5s |    5        20       15       25    
  6s |    18       30       18       30    
 Max |    30       28       26       30    
 Min |    7        10       7        7     
   T |    38       35       35       32    
   K |    50       45       50       50    
   F |    66       64       60       59    
   P |    70       70       74       66    
   Y |    80       85       90       85    
      ------------------------------------
🏁 FINAL SCORE: 1991

```
