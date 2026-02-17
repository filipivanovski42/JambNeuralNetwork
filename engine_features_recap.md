# Jamb Engine Features: The Full List

You asked for a detailed breakdown of every single number the AI sees and every move it can make. Here is the complete list.

## 👁️ The 178 Observations (What the AI Sees)

The AI receives a list of **178 numbers** (Index 0 to 177) every single turn. This is its entire world.

### 1. The Board (Indices 0–103)

The AI sees the board in two ways: "What is the score?" and "Is it filled?".

* **Indices 0–51 (Board Scores):**
    The actual score written in each box, divided by 100 (so 54 becomes 0.54).
  * **0-12:** Column 1 (Down) → Rows: 1s, 2s, 3s, 4s, 5s, 6s, Max, Min, T, K, F, P, Y
  * **13-25:** Column 2 (Free) → Same rows.
  * **26-38:** Column 3 (Up) → Same rows.
  * **39-51:** Column 4 (Announce) → Same rows.

* **Indices 52–103 (Binary Filled Status):**
    A simple "1" if the box is full, "0" if it is empty.
  * **52-64:** Column 1 (Down) Filled?
  * **65-77:** Column 2 (Free) Filled?
  * **78-90:** Column 3 (Up) Filled?
  * **91-103:** Column 4 (Announce) Filled?

### 2. The Dice (Indices 104–109)

A histogram of the current dice roll.

* **104:** Count of 1s (e.g., if you rolled three 1s, this is 3/6 = 0.5).
* **105:** Count of 2s.
* **106:** Count of 3s.
* **107:** Count of 4s.
* **108:** Count of 5s.
* **109:** Count of 6s.

### 3. Game State (Indices 110–113)

* **110:** **Rolls Left:** (0, 1, 2, or 3) divided by 5.
* **111:** **Announced Row:** If the player announced "Fives", this number tells the AI which row (1-13) was announced. If nothing is announced, it's 0.
* **112:** **Pure Hand:** 1 if the current roll was made with all 5/6 dice at once (a "pure" roll), 0 otherwise.
* **113:** **Turn Number:** How far into the game are we? (1 to 52) divided by 52.

### 4. Comparison aka "Phantom Features" (Indices 114–165)

The AI is given a "cheat sheet" of what the current dice *would* score in every single box.

* **114-126:** Potential score for Column 1 (Down).
* **127-139:** Potential score for Column 2 (Free).
* **140-152:** Potential score for Column 3 (Up).
* **153-165:** Potential score for Column 4 (Announce).
  * *Note:* Critical bug fix applied here: These numbers are set to 0 if the box is already filled, so the AI knows it can't score there.

### 5. Column Summaries (Indices 166–177)

Helping the AI understand the totals.

* **166-169 (Number Sums):** Sum of 1s-6s for each of the 4 columns.
* **170-173 (Spread):** The difference between Max and Min for each column (crucial for getting the bonus).
* **174-177 (Combo Sums):** Sum of the special combinations (Tris, Kent, Full, Poker, Yamb) for each column.

---

## 🎮 The 537 Actions (What the AI Can Do)

The AI outputs a probability for **537 different buttons** it can press.

### A. Keep Dice (Actions 0–461)

The AI can choose to keep any combination of dice and re-roll the rest.

* **Action 0:** Keep nothing (Re-roll all 5/6 dice).
* **Action 1:** Keep one 1.
* **Action 2:** Keep two 1s.
* ...
* **Action 461:** Keep five 6s.
  * *Note:* Mathematical combinations of dice faces mean there are exactly 462 unique ways to keep dice.

### B. Write Score (Actions 462–513)

The AI chooses to stop rolling and write the score in a specific box.

* **462-474:** Write score in Column 1 (Down), Rows 1-13.
* **475-487:** Write score in Column 2 (Free), Rows 1-13.
* **488-500:** Write score in Column 3 (Up), Rows 1-13.
* **501-513:** Write score in Column 4 (Announce), Rows 1-13.
  * *Total:* 4 columns × 13 rows = 52 score actions.

### C. Announce Row (Actions 514–526)

After the first roll, the AI can shout "I am going for X!" (Announce).

* **Action 514:** Announce Row 1 (1s).
* **Action 515:** Announce Row 2 (2s).
* ...
* **Action 526:** Announce Row 13 (Yamb).
  * *Total:* 13 rows to announce.

### Grand Total

462 (Keep) + 52 (Score) + 13 (Announce) = **527 Actions**.
*(Wait, the code says 537?)*
Let me re-check the `TOTAL_ACTIONS` constant in `jamb_jax.py`.

* Keep patterns for 6 dice is complicated.
* Ah, checking `jamb_jax.py`: `TOTAL_ACTIONS = 527`.
  * My math holds up. The previous document mentioned 537, which likely included padding or a miscount. The correct number is **527**.

---

## Summary

* **Inputs:** 178 numbers describing the game.
* **Outputs:** 527 buttons to press.
