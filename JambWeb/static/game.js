const state = {
    gameId: null,
    board: null,
    dice: [],
    kept: [false, false, false, false, false, false],
    rollsLeft: 3,
    turn: 1,
    announcedRow: -1,
    engineEnabled: true,
    isEditing: false,
    originalDice: []
};

const ROWS = ["1s", "2s", "3s", "4s", "5s", "6s", "Max", "Min", "Trips", "Straight", "Full", "Poker", "Yamb"];
const COLS = ["Down", "Free", "Up", "Anno"];

document.addEventListener('DOMContentLoaded', () => {
    initTable();
    setupEventListeners();
    startNewGame();
});

function initTable() {
    const tbody = document.getElementById('scoreBody');
    tbody.innerHTML = '';

    const createRow = (r) => {
        const tr = document.createElement('tr');
        const th = document.createElement('td');
        th.innerText = ROWS[r];
        th.style.fontWeight = 'bold';
        tr.appendChild(th);
        COLS.forEach((_, c) => {
            const td = document.createElement('td');
            td.id = `cell-${r}-${c}`;
            td.dataset.r = r;
            td.dataset.c = c;
            td.addEventListener('click', () => handleCellClick(r, c));
            tr.appendChild(td);
        });
        return tr;
    };

    const createSubtotalRow = (label, idPrefix) => {
        const tr = document.createElement('tr');
        tr.className = 'subtotal-row';
        const th = document.createElement('td');
        th.innerText = label;
        tr.appendChild(th);
        for (let c = 0; c < 4; c++) {
            const td = document.createElement('td');
            td.id = `${idPrefix}-${c}`;
            td.innerText = '0';
            tr.appendChild(td);
        }
        return tr;
    };

    const createSpacer = () => {
        const tr = document.createElement('tr');
        tr.className = 'section-spacer';
        const td = document.createElement('td');
        td.colSpan = 5;
        tr.appendChild(td);
        return tr;
    };

    // Section 1
    for (let i = 0; i <= 5; i++) tbody.appendChild(createRow(i));
    tbody.appendChild(createSubtotalRow("Sum (1-6) + 30", "sub1"));
    tbody.appendChild(createSpacer());

    // Section 2
    for (let i = 6; i <= 7; i++) tbody.appendChild(createRow(i));
    tbody.appendChild(createSubtotalRow("(M-m)*1", "sub2"));
    tbody.appendChild(createSpacer());

    // Section 3
    for (let i = 8; i <= 12; i++) tbody.appendChild(createRow(i));
    tbody.appendChild(createSubtotalRow("Sum Combos", "sub3"));
    tbody.appendChild(createSpacer());

    // Footer
    const tfoot = document.querySelector('#scoreTable tfoot');
    tfoot.innerHTML = '';

    const totalRow = document.createElement('tr');
    totalRow.className = 'grand-total-row';
    const thTotal = document.createElement('td');
    thTotal.innerText = 'Total';
    totalRow.appendChild(thTotal);
    for (let c = 0; c < 4; c++) {
        const td = document.createElement('td');
        td.id = `total-${c}`;
        td.innerText = '0';
        totalRow.appendChild(td);
    }
    tfoot.appendChild(totalRow);

    const grandRow = document.createElement('tr');
    grandRow.className = 'grand-total-row';
    const tdGrand = document.createElement('td');
    tdGrand.colSpan = 5;
    tdGrand.style.fontSize = '1.3rem';
    tdGrand.innerHTML = 'Grand Total: <span id="grandTotal">0</span>';
    grandRow.appendChild(tdGrand);
    tfoot.appendChild(grandRow);
}

function setupEventListeners() {
    document.getElementById('newGameBtn').addEventListener('click', startNewGame);
    document.getElementById('rollBtn').addEventListener('click', handleRoll);
    document.getElementById('engineToggle').addEventListener('change', (e) => {
        state.engineEnabled = e.target.checked;
        document.getElementById('analysisPanel').classList.toggle('hidden', !state.engineEnabled);
    });

    const modal = document.getElementById('rulesModal');
    const btn = document.getElementById('rulesBtn');
    const span = document.getElementsByClassName('close-btn')[0];
    if (btn) btn.onclick = () => modal.style.display = "block";
    if (span) span.onclick = () => modal.style.display = "none";
    window.onclick = (e) => { if (e.target == modal) modal.style.display = "none"; };

    document.getElementById('editDiceBtn').addEventListener('click', () => toggleEditMode(false));
    document.getElementById('saveDiceBtn').addEventListener('click', saveDice);
}

function toggleEditMode(save = false) {
    const diceArea = document.getElementById('diceArea');
    const rollBtn = document.getElementById('rollBtn');
    const saveBtn = document.getElementById('saveDiceBtn');
    const editBtn = document.getElementById('editDiceBtn');

    if (!state.isEditing) {
        // Enter Edit Mode
        state.isEditing = true;
        state.originalDice = [...state.dice]; // Snapshot

        diceArea.classList.add('editing');
        rollBtn.style.display = 'none';
        saveBtn.style.display = 'inline-block';
        editBtn.innerText = 'Cancel Edit';
    } else {
        // Exit Edit Mode
        state.isEditing = false;
        if (!save) {
            // Restore original dice if not saving
            state.dice = [...state.originalDice];
        }

        diceArea.classList.remove('editing');
        rollBtn.style.display = 'inline-block';
        saveBtn.style.display = 'none';
        editBtn.innerText = 'Edit Dice';
    }
    renderDice();
}

async function saveDice() {
    if (!state.isEditing) return;
    const res = await fetch('/api/set_dice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: state.gameId, dice: state.dice })
    });
    const data = await res.json();
    if (data.error) {
        alert("Error saving dice: " + data.error);
        return;
    }
    // Exit edit mode, saving changes (no revert)
    toggleEditMode(true);
    updateState(data);
}

async function startNewGame() {
    const res = await fetch('/api/start', { method: 'POST' });
    const data = await res.json();
    state.gameId = data.game_id;
    updateState(data);
}

async function handleRoll() {
    if (state.rollsLeft <= 0) {
        alert("No rolls left! You must score.");
        return;
    }
    await sendAction({ kept: state.kept });
}

function handleCellClick(r, c) {
    if (state.isEditing) return;
    let actionId = -1;
    if (c === 3 && state.rollsLeft === 2 && state.announcedRow === -1) {
        actionId = 514 + r;
    } else {
        actionId = 462 + (r * 4 + c);
    }
    sendAction(actionId);
}

async function sendAction(payload) {
    const body = { game_id: state.gameId };
    if (typeof payload === 'number') body.action = payload;
    else Object.assign(body, payload);

    const res = await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    const data = await res.json();
    if (data.error) {
        if (data.error.includes("Illegal move")) alert("Invalid Move!");
        else alert(data.error);
        return;
    }
    updateState(data);
}

function updateState(data) {
    const s = data.state;
    state.board = s.board;
    state.dice = s.dice;
    state.rollsLeft = s.rolls_left;
    state.announcedRow = s.announced_row;
    state.kept = new Array(6).fill(false);

    renderDice();
    renderBoard();
    updateAnalysis(data.analysis);
}

function renderDice() {
    const diceCont = document.getElementById('diceContainer');
    diceCont.innerHTML = '';
    state.dice.forEach((val, i) => {
        const d = document.createElement('div');
        d.className = 'die';
        if (state.isEditing) d.classList.add('editing');

        d.innerText = val === 0 ? '?' : val;
        if (val === 0) d.classList.add('placeholder');
        else if (state.kept[i] && !state.isEditing) d.classList.add('kept');

        d.addEventListener('click', () => {
            if (state.isEditing) {
                // FIXED: Read current value from state, not stale closure 'val'
                let currentVal = state.dice[i];
                let newVal = currentVal + 1;
                if (newVal > 6) newVal = 1;
                state.dice[i] = newVal;
                d.innerText = newVal;
            } else {
                if (val > 0) {
                    state.kept[i] = !state.kept[i];
                    d.classList.toggle('kept', state.kept[i]);
                }
            }
        });
        diceCont.appendChild(d);
    });
    document.getElementById('rollsLeft').innerText = state.rollsLeft;
}

function renderBoard() {
    ROWS.forEach((_, r) => {
        COLS.forEach((_, c) => {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (!cell) return;
            const val = state.board[r][c];
            if (val > -1) {
                cell.innerText = val;
                cell.classList.add('filled');
            } else {
                cell.innerText = '';
                cell.classList.remove('filled');
            }
        });
    });

    const totals = { sub1: [0, 0, 0, 0], sub2: [0, 0, 0, 0], sub3: [0, 0, 0, 0], col: [0, 0, 0, 0], grand: 0 };

    for (let c = 0; c < 4; c++) {
        let sum1 = 0;
        for (let r = 0; r <= 5; r++) { if (state.board[r][c] > -1) sum1 += state.board[r][c]; }
        if (sum1 >= 60) sum1 += 30;
        totals.sub1[c] = sum1;

        let valMax = state.board[6][c], valMin = state.board[7][c], valOnes = state.board[0][c];
        let res2 = 0;
        if (valMax > -1 && valMin > -1 && valOnes > -1) res2 = (valMax - valMin) * valOnes;
        totals.sub2[c] = res2;

        let sum3 = 0;
        for (let r = 8; r <= 12; r++) { if (state.board[r][c] > -1) sum3 += state.board[r][c]; }
        totals.sub3[c] = sum3;

        totals.col[c] = totals.sub1[c] + totals.sub2[c] + totals.sub3[c];
        totals.grand += totals.col[c];
    }

    for (let c = 0; c < 4; c++) {
        const el1 = document.getElementById(`sub1-${c}`);
        if (el1) el1.innerText = totals.sub1[c];
        const el2 = document.getElementById(`sub2-${c}`);
        if (el2) el2.innerText = totals.sub2[c];
        const el3 = document.getElementById(`sub3-${c}`);
        if (el3) el3.innerText = totals.sub3[c];
        const elt = document.getElementById(`total-${c}`);
        if (elt) elt.innerText = totals.col[c];
    }

    document.getElementById('grandTotal').innerText = totals.grand;
}

function updateAnalysis(analysis) {
    const list = document.getElementById('suggestionsList');
    list.innerHTML = '';
    if (state.engineEnabled && analysis) {
        analysis.forEach(item => {
            const div = document.createElement('div');
            div.className = 'suggestion-item';
            div.innerHTML = `<span class="action">${item.name}</span><span class="prob">${(item.prob * 100).toFixed(1)}%</span>`;
            list.appendChild(div);
        });
    }
}
