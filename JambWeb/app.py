from flask import Flask, render_template, request, jsonify, session
import jax
import jax.numpy as jnp
import numpy as np
import os
import secrets
# Import necessary components including KEEP_PATTERNS
from game_logic import step, reset, get_obs, get_action_mask, calculate_total_score, NUM_KEEP_ACTIONS, NUM_SCORE_ACTIONS, KEEP_PATTERNS
from agent import Agent

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load Agent
MODEL_PATH = "model.npz"
agent = None

import sys

# ...

try:
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Attempting to load agent from {os.path.abspath(MODEL_PATH)}...", file=sys.stderr)
        agent = Agent(MODEL_PATH)
        print("✅ Agent loaded successfully", file=sys.stderr)
    else:
        print(f"⚠️ Model file not found at {os.path.abspath(MODEL_PATH)}, running without suggestions", file=sys.stderr)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Error loading agent: {e}", file=sys.stderr)

@app.route('/')
def index():
    return render_template('index.html')

GAMES = {}

def serialize_state(state):
    # Reconstruct dice from histogram
    # state.dice_hist is (6,) array of counts per face (1-6)
    dice = []
    hist = np.array(state.dice_hist)
    for i in range(6):
        count = int(hist[i])
        dice.extend([i+1] * count)
    
    # Pad with 0 if less than 6
    while len(dice) < 6:
        dice.append(0)

    return {
        'board': np.array(state.board).tolist(),
        'dice': dice, 
        'dice_hist': np.array(state.dice_hist).tolist(),
        'rolls_left': int(state.rolls_left),
        'turn': int(state.turn_number),
        'announced_row': int(state.announced_row),
        'game_over': bool(state.game_over),
        'mask': np.array(get_action_mask(state)).tolist()
    }

def get_analysis(state):
    if not agent: return None
    # Get top 5 actions
    probs, values = agent.predict(state)
    return agent.decode_predictions(probs, state)

@app.route('/api/start', methods=['POST'])
def start():
    game_id = secrets.token_hex(8)
    seed = secrets.randbits(32)
    key = jax.random.PRNGKey(seed)
    state, obs = reset(key)
    
    GAMES[game_id] = {
        'state': state,
        'key': key,
        'history': []
    }
    
    return jsonify({
        'game_id': game_id,
        'state': serialize_state(state),
        'analysis': get_analysis(state)
    })

@app.route('/api/action', methods=['POST'])
def action():
    data = request.json
    game_id = data.get('game_id')
    
    if game_id not in GAMES:
        return jsonify({'error': 'Game not found'}), 404
        
    game = GAMES[game_id]
    state = game['state']
    key = game['key']
    
    action_idx = -1

    if 'kept' in data:
        # Resolve Keep Action
        kept_mask = data['kept'] # [True, False, ...]
        
        # 1. Reconstruct current dice exactly as sent to frontend
        # This MUST MATCH serialize_state logic
        current_dice = []
        hist = np.array(state.dice_hist)
        for i in range(6):
            current_dice.extend([i+1] * int(hist[i]))
        while len(current_dice) < 6: current_dice.append(0)
        
        # 2. Filter kept values using the mask
        if len(kept_mask) != 6:
            # Fallback if frontend sends different length
            return jsonify({'error': 'Invalid kept mask length'}), 400
            
        kept_values = []
        for d, k in zip(current_dice, kept_mask):
            if k and d > 0: # Only keep non-zero dice
                kept_values.append(d)
        
        # 3. Find Strategy Index
        # Find Strategy Index
        # Convert kept values to histogram format used by KEEP_PATTERNS
        # KEEP_PATTERNS is (462, 6) where each row is [count_1s, count_2s, ..., count_6s]
        target_hist = [0] * 6
        for v in kept_values:
            if 1 <= v <= 6:
                target_hist[v-1] += 1
        
        target_arr = jnp.array(target_hist, dtype=jnp.int8)
        
        # Search in KEEP_PATTERNS
        matches = jnp.all(KEEP_PATTERNS == target_arr, axis=1)
        
        if not jnp.any(matches):
             return jsonify({'error': f'Invalid keep pattern: {target}'}), 400
             
        action_idx = int(jnp.argmax(matches))
            
    else:
        # Score/Anno action directly
        action_idx = data.get('action') 
        if action_idx is None:
             return jsonify({'error': 'No action provided'}), 400

    # Validate Move with Mask
    mask = get_action_mask(state)
    if not mask[action_idx]:
         return jsonify({'error': f'Illegal move for action {action_idx}'}), 400

    # Step
    key, subkey = jax.random.split(key)
    next_state, next_obs, reward, done, _ = step(subkey, state, action_idx)
    
    game['state'] = next_state
    game['key'] = key
    
    analysis = None
    try:
        if not done:
            analysis = get_analysis(next_state)
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
    
    return jsonify({
        'game_id': game_id,
        'state': serialize_state(next_state),
        'analysis': analysis,
        'score': int(calculate_total_score(next_state.board))
    })

@app.route('/api/set_dice', methods=['POST'])
def set_dice():
    data = request.json
    game_id = data.get('game_id')
    dice_values = data.get('dice') # List of ints [1..6]
    
    if game_id not in GAMES:
        return jsonify({'error': 'Game not found'}), 404
    
    # Validate input
    if not isinstance(dice_values, list) or len(dice_values) > 6:
        return jsonify({'error': 'Invalid dice list'}), 400
        
    # Convert to histogram
    new_hist = [0] * 6
    for d in dice_values:
        if 1 <= d <= 6:
            new_hist[d-1] += 1
            
    # Update state
    game = GAMES[game_id]
    current_state = game['state']
    
    # Create new parsed state
    new_state = current_state._replace(
        dice_hist=jnp.array(new_hist, dtype=jnp.int8)
    )
    
    game['state'] = new_state
    
    # Get analysis for new dice
    analysis = None
    try:
        analysis = get_analysis(new_state)
    except Exception as e:
        print(f"❌ Analysis Error (Set Dice): {e}")

    return jsonify({
        'game_id': game_id,
        'state': serialize_state(new_state),
        'analysis': analysis,
        'score': int(calculate_total_score(new_state.board))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
